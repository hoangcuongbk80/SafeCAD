"""Geometry and extraction tests use deterministic backbones, without downloads."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safecad.vision import FrozenEncoder, ImageProcessor


class DinoStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def get_intermediate_layers(self, x, n, reshape, norm):
        self.calls.append((n, reshape, norm))
        m = torch.nn.functional.avg_pool2d(x, 14)
        return (m.repeat(1, 256, 1, 1), (m+0.1).repeat(1, 256, 1, 1))


def test_dino_blocks_layernorm_features_and_four_views():
    stub = DinoStub()
    processor = ImageProcessor(FrozenEncoder(model=stub))
    image = np.full((96, 160, 3), 128, dtype=np.uint8)
    features, geometry = processor.extract(image)
    assert len(stub.calls) == 5
    assert stub.calls[0] == ([7, 11], True, True)
    assert features.z.shape[1] == 1536
    np.testing.assert_allclose(np.linalg.norm(features.z, axis=1), 1, atol=1e-6)
    assert features.global_consistency >= 0.95
    assert np.all(features.patch_consistency >= 0.95)
    assert (features.coords >= 0).all() and (features.coords <= 1).all()
    assert len(features.z) < 1024
    assert geometry.original_hw == (96, 160)
    map_ = processor.anomaly_map(np.ones(len(features.z), np.float32)*0.4, geometry)
    assert map_.shape == (96, 160)
    np.testing.assert_allclose(map_, 0.4, atol=1e-6)


def test_map_preserves_horizontal_score_order_after_unpadding():
    processor = ImageProcessor(FrozenEncoder(model=DinoStub()))
    features, geometry = processor.extract(np.zeros((120, 60, 3), np.uint8), consistency=False)
    assert features.patch_consistency is None
    map_ = processor.anomaly_map(features.coords[:, 0], geometry)
    assert map_.shape == (120, 60)
    assert np.all(np.diff(map_, axis=1) >= -1e-7)
    assert map_[60, 0] < map_[60, -1]


def test_compact_actual_architecture_without_weight_download():
    torchvision = pytest.importorskip("torchvision")
    model = torchvision.models.mobilenet_v3_small(weights=None)
    encoder = FrozenEncoder("compact", model=model)
    features, geometry = ImageProcessor(encoder).extract(np.full((256, 256, 3), 128, np.uint8), consistency=False)
    assert features.z.shape == (1024, 72)
    assert geometry.grid_hw == (32, 32)
    assert not encoder.model.training
    assert all(not p.requires_grad for p in encoder.model.parameters())


def test_extreme_aspect_ratio_fails_explicitly():
    processor = ImageProcessor(FrozenEncoder(model=DinoStub()))
    with pytest.raises(ValueError, match="no fully valid"):
        processor.extract(np.zeros((1, 1000, 3), np.uint8), consistency=False)
