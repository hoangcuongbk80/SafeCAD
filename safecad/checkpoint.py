"""Portable JSON + NumPy checkpoints, without executable pickle payloads."""
from dataclasses import fields, is_dataclass
from io import BytesIO
import json
from pathlib import Path
import os
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
from .config import Config
from .core import SafeCAD, Domain, Observation, Prototype


def save(model, path):
    arrays = {}

    def encode(value):
        if isinstance(value, np.ndarray):
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if is_dataclass(value):
            return {"type": type(value).__name__, "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
        if isinstance(value, set):
            return {"set": sorted(value)}
        if isinstance(value, list):
            return [encode(v) for v in value]
        if isinstance(value, dict):
            return {k: encode(v) for k, v in value.items()}
        if isinstance(value, np.generic):
            return value.item()
        return value

    state = {k: encode(v) for k, v in model.__dict__.items() if k != "event_sink"}
    payload = BytesIO()
    np.savez_compressed(payload, **arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with ZipFile(temp, "w", compression=ZIP_DEFLATED) as z:
            z.writestr("state.json", json.dumps({"version": 1, "state": state}, allow_nan=False))
            z.writestr("arrays.npz", payload.getvalue())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load(path, event_sink=None):
    classes = {t.__name__: t for t in (Config, Domain, Observation, Prototype)}
    with ZipFile(path) as z:
        envelope = json.loads(z.read("state.json"))
        if envelope["version"] != 1:
            raise ValueError("unsupported checkpoint version")
        with np.load(BytesIO(z.read("arrays.npz")), allow_pickle=False) as npz:
            arrays = {k: npz[k].copy() for k in npz.files}

    def decode(value):
        if isinstance(value, list):
            return [decode(v) for v in value]
        if isinstance(value, dict):
            if set(value) == {"array"}:
                return arrays[value["array"]]
            if set(value) == {"set"}:
                return set(value["set"])
            if set(value) == {"type", "fields"}:
                return classes[value["type"]](**{k: decode(v) for k, v in value["fields"].items()})
            return {k: decode(v) for k, v in value.items()}
        return value

    model = SafeCAD(event_sink=event_sink)
    model.__dict__.update({k: decode(v) for k, v in envelope["state"].items() if k != "event_sink"})
    for state in model.domains.values():
        state.anchor.setflags(write=False)
        state.initial_context.setflags(write=False)
    return model
