"""Feature validation, exact search, fixed patch retention and robust pooling."""
from dataclasses import dataclass
import math
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def context(features, trim_fraction):
    n = int(len(features) * trim_fraction)
    ordered = np.sort(features, axis=0)
    return normalize(ordered[n:len(features)-n].mean(axis=0))


def distances(a, b):
    # Direct differences avoid cancellation for identical and very close vectors.
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)


def nearest(query, reference, chunk=128):
    out = np.full(len(query), np.inf, dtype=np.float32)
    for i in range(0, len(query), chunk):
        for k in range(0, len(reference), chunk):
            out[i:i+chunk] = np.minimum(out[i:i+chunk],
                distances(query[i:i+chunk], reference[k:k+chunk]).min(axis=1))
    return out


def coreset_indices(features, capacity, keys=None):
    """Oldest-first greedy farthest-point sampling, deterministic origin ties."""
    if len(features) <= capacity:
        return list(range(len(features)))
    order = sorted(range(len(features)), key=lambda i: keys[i] if keys else i)
    chosen = [order[0]]
    best = np.full(len(features), np.inf)
    for _ in range(capacity - 1):
        best = np.minimum(best, np.linalg.norm(features - features[chosen[-1]], axis=1))
        best[chosen] = -np.inf
        chosen.append(max(order, key=lambda i: best[i]))
    return chosen


def top_mean(scores, fraction):
    m = max(1, math.ceil(fraction * len(scores)))
    return float(np.partition(scores, len(scores)-m)[-m:].mean())


def retained_indices(coords, rows, cols):
    selected = []
    cells = np.minimum((coords * [cols, rows]).astype(int), [cols-1, rows-1])
    for row in range(rows):
        for col in range(cols):
            candidates = np.flatnonzero(np.all(cells == [col, row], axis=1))
            if len(candidates):
                center = np.array([(col+0.5)/cols, (row+0.5)/rows])
                selected.append(int(candidates[np.argmin(np.linalg.norm(coords[candidates]-center, axis=1))]))
    return np.array(selected, dtype=int)


@dataclass
class Features:
    """Valid, row-major inspection patches, with coordinates in (x, y) order.

    Consistency is required for adaptation. Feature-only callers must supply
    measured statistics rather than silently assuming perfectly stable views.
    """
    z: np.ndarray
    coords: np.ndarray
    global_consistency: float | None = None
    patch_consistency: np.ndarray | None = None

    def __post_init__(self):
        self.z = np.array(self.z, dtype=np.float32, copy=True)
        self.coords = np.array(self.coords, dtype=np.float32, copy=True)
        if self.z.ndim != 2 or min(self.z.shape) < 1 or not np.isfinite(self.z).all():
            raise ValueError("z must be a finite nonempty [patches, channels] array")
        if self.coords.shape != (len(self.z), 2) or not np.isfinite(self.coords).all():
            raise ValueError("coords must be finite [patches, 2]")
        if np.any((self.coords < 0) | (self.coords > 1)):
            raise ValueError("coordinates must lie in [0, 1]")
        norms = np.linalg.norm(self.z, axis=1)
        if not np.all(np.isclose(norms, 1, atol=1e-5) | (norms <= 1e-12)):
            raise ValueError("patch features must be L2 normalized (or zero)")
        if self.global_consistency is not None and not 0 <= self.global_consistency <= 1:
            raise ValueError("global consistency must lie in [0, 1]")
        if self.patch_consistency is not None:
            self.patch_consistency = np.array(self.patch_consistency, dtype=np.float32, copy=True)
            if self.patch_consistency.shape != (len(self.z),) or not np.isfinite(self.patch_consistency).all():
                raise ValueError("patch consistency must be finite [patches]")
            if np.any((self.patch_consistency < 0) | (self.patch_consistency > 1)):
                raise ValueError("patch consistency must lie in [0, 1]")

    def descriptor(self, trim_fraction=0.1):
        return context(self.z, trim_fraction)
