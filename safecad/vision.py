"""Frozen DINOv2 / MobileNetV3-Small features and image-map geometry.

Torch is optional for the feature-level QCM API. Both DINOv2 intermediate
blocks are extracted together with checkpoint output LayerNorm enabled.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from .features import Features, context


class FrozenEncoder:
    def __init__(self, kind="dinov2", device="cpu", repository=None, checkpoint=None, model=None):
        import torch
        self.torch = torch
        self.kind, self.device = kind, torch.device(device)
        if kind not in ("dinov2", "compact"):
            raise ValueError("encoder must be dinov2 or compact")
        self.input_size = 448 if kind == "dinov2" else 256
        self.patch_size = 14 if kind == "dinov2" else 8
        if model is None:
            if kind == "dinov2":
                model = torch.hub.load(str(repository) if repository else "facebookresearch/dinov2",
                    "dinov2_vitb14", source="local" if repository else "github",
                    pretrained=checkpoint is None, trust_repo=True)
            else:
                from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
                model = mobilenet_v3_small(weights=None if checkpoint else MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            if checkpoint:
                model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model = model.eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def __call__(self, images):
        torch = self.torch
        import torch.nn.functional as F
        with torch.inference_mode():
            if self.kind == "dinov2":
                maps = self.model.get_intermediate_layers(images, n=[7, 11], reshape=True, norm=True)
            else:
                maps = []
                out = images
                for index, layer in enumerate(self.model.features):
                    out = layer(out)
                    if index in (3, 8):
                        maps.append(out)
                    if index == 8:
                        break
            size = maps[0].shape[-2:]
            maps = [F.normalize(F.interpolate(m, size=size, mode="bilinear", align_corners=False)
                    if m.shape[-2:] != size else m, dim=1, eps=1e-12) for m in maps]
            return F.normalize(torch.cat(maps, dim=1), dim=1, eps=1e-12)


@dataclass
class Geometry:
    original_hw: tuple
    resized_hw: tuple
    top: int
    left: int
    canvas: int
    grid_hw: tuple
    valid_indices: np.ndarray
    coords: np.ndarray


class ImageProcessor:
    def __init__(self, encoder: FrozenEncoder, trim_fraction=0.1):
        self.encoder = encoder
        self.trim_fraction = trim_fraction

    def _image(self, image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255
        elif isinstance(image, Image.Image):
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        else:
            arr = np.asarray(image)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255
            else:
                arr = arr.astype(np.float32)
        if arr.ndim != 3 or arr.shape[2] != 3 or min(arr.shape[:2]) == 0 or not np.isfinite(arr).all():
            raise ValueError("image must be finite RGB [height, width, 3]")
        if np.any((arr < 0) | (arr > 1)):
            raise ValueError("float RGB images must lie in [0, 1]")
        return arr

    def extract(self, image, consistency=True):
        import torch
        import torch.nn.functional as F
        arr = self._image(image)
        h, w = arr.shape[:2]
        size = self.encoder.input_size
        scale = min(size/h, size/w)
        rh, rw = max(1, round(h*scale)), max(1, round(w*scale))
        top, left = (size-rh)//2, (size-rw)//2
        tensor = torch.from_numpy(arr.copy()).permute(2, 0, 1)[None].to(self.encoder.device)
        tensor = F.interpolate(tensor, size=(rh, rw), mode="bilinear", align_corners=False)
        views = [tensor]
        if consistency:
            # Four separate transforms, two brightness and two contrast views.
            gray_mean = (tensor * tensor.new_tensor([0.2989, 0.587, 0.114])[None, :, None, None]).sum(dim=1, keepdim=True).mean()
            views += [(tensor*f).clamp(0, 1) for f in (0.95, 1.05)]
            views += [((tensor-gray_mean)*f+gray_mean).clamp(0, 1) for f in (0.95, 1.05)]
        mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = tensor.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        maps = []
        # Sequential views keep transient encoder memory bounded on edge devices.
        for view in views:
            padded = mean.expand(1, 3, size, size).clone()
            padded[:, :, top:top+rh, left:left+rw] = view
            maps.append(self.encoder((padded-mean)/std)[0].permute(1, 2, 0).cpu().numpy())
        gh, gw = maps[0].shape[:2]
        cell_h, cell_w = size/gh, size/gw
        yy, xx = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        # Exclude any patch whose grid cell intersects preprocessing padding.
        valid = ((xx*cell_w >= left) & ((xx+1)*cell_w <= left+rw)
                 & (yy*cell_h >= top) & ((yy+1)*cell_h <= top+rh))
        ids = np.flatnonzero(valid.ravel())
        if not len(ids):
            raise ValueError("aspect ratio leaves no fully valid inspection patches")
        coords = np.stack([((xx+0.5)*cell_w-left)/rw, ((yy+0.5)*cell_h-top)/rh], axis=-1).reshape(-1, 2)[ids]
        patches = [m.reshape(-1, m.shape[-1])[ids] for m in maps]
        cg, cp = None, None
        if consistency:
            g = context(patches[0], self.trim_fraction)
            cg = float(np.clip(1-np.mean([np.linalg.norm(g-context(z, self.trim_fraction)) for z in patches[1:]])/2, 0, 1))
            cp = np.clip(1-np.mean([np.linalg.norm(patches[0]-z, axis=1) for z in patches[1:]], axis=0)/2, 0, 1)
        return Features(patches[0], coords, cg, cp), Geometry((h, w), (rh, rw), top, left, size, (gh, gw), ids, coords)

    @staticmethod
    def anomaly_map(scores, geometry: Geometry):
        import torch
        import torch.nn.functional as F
        gh, gw = geometry.grid_hw
        if len(scores) != len(geometry.valid_indices):
            raise ValueError("score count differs from inspection grid")
        # Replicate the nearest valid row/column into excluded border cells so
        # interpolation does not mix artificial zero scores into inspection pixels.
        rows, cols = np.unravel_index(geometry.valid_indices, (gh, gw))
        grid = np.empty((gh, gw), np.float32)
        grid[rows, cols] = scores
        row_ids = np.clip(np.arange(gh), rows.min(), rows.max())
        col_ids = np.clip(np.arange(gw), cols.min(), cols.max())
        grid = grid[row_ids[:, None], col_ids[None, :]]
        tensor = torch.from_numpy(grid.copy())[None, None]
        full = F.interpolate(tensor, size=(geometry.canvas, geometry.canvas), mode="bilinear", align_corners=False)
        rh, rw = geometry.resized_hw
        cropped = full[:, :, geometry.top:geometry.top+rh, geometry.left:geometry.left+rw]
        result = F.interpolate(cropped, size=geometry.original_hw, mode="bilinear", align_corners=False)
        return result[0, 0].numpy()
