"""SafeCAD memory, support, routing and calibration configuration."""
from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class Config:
    anchor_capacity: int = 1024
    pending_capacity: int = 256
    active_capacity: int = 768
    quarantine_capacity: int = 64
    unsupported_capacity: int = 64
    max_domains: int = 32
    layout_rows: int = 8
    layout_cols: int = 8
    episode_cap: int = 1
    retention_stride: int = 1
    episodes_per_window: int = 4
    context_episodes: int = 3
    patch_episodes: int = 3
    activation_episodes: int = 3
    context_windows: int = 2
    patch_windows: int = 2
    activation_windows: int = 2
    quarantine_lifetime: int = 8
    pending_lifetime: int = 8
    unsupported_lifetime: int = 8
    context_radius: float = 0.20
    patch_radius: float = 0.15
    duplicate_radius: float = 0.05
    spatial_radius: float = 0.10
    route_radius: float = 0.35
    context_consistency: float = 0.95
    patch_consistency: float = 0.95
    context_eta: float = 0.05
    trim_fraction: float = 0.10
    top_fraction: float = 0.01
    alpha: float = 0.05
    calibration_min: int = 20
    distance_chunk: int = 128

    def __post_init__(self):
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type == int or f.type == "int":
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{f.name} must be a positive integer")
            elif not math.isfinite(value):
                raise ValueError(f"{f.name} must be finite")
        for name in ("context_radius", "patch_radius", "duplicate_radius", "spatial_radius", "route_radius"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.duplicate_radius > self.patch_radius:
            raise ValueError("duplicate_radius must not exceed patch_radius")
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        for name in ("top_fraction", "context_eta"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in ("context_consistency", "patch_consistency"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.activation_windows > self.pending_lifetime:
            raise ValueError("activation windows exceed pending lifetime")
        if self.activation_episodes > self.episodes_per_window * self.pending_lifetime:
            raise ValueError("activation episode requirement is infeasible")

    @classmethod
    def compact(cls, **overrides):
        values = dict(context_radius=0.30, patch_radius=0.22, duplicate_radius=0.07)
        values.update(overrides)
        return cls(**values)
