"""End-to-end local-image replay with a real compact backbone, no downloads."""
import argparse
import json
import numpy as np
import pytest
from PIL import Image
from safecad import SafeCAD


def test_commission_stream_and_frozen_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    from safecad.cli import commission, stream
    torch.manual_seed(0)
    torch.set_num_threads(2)
    weights = tmp_path/"weights.pt"
    torch.save(torchvision.models.mobilenet_v3_small(weights=None).state_dict(), weights)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (80, 120, 3), dtype=np.uint8)
    for i in range(4):
        Image.fromarray(image).save(tmp_path/f"image_{i}.png")
    manifest = tmp_path/"commission.json"
    manifest.write_text(json.dumps({"domains": [{"id": "camera", "initialization": ["image_0.png", "image_1.png"],
        "calibration": ["image_2.png", "image_3.png"]}]}))
    config = tmp_path/"config.json"
    config.write_text(json.dumps(dict(calibration_min=2, anchor_capacity=4, layout_rows=1, layout_cols=1)))
    checkpoint = tmp_path/"initial.safecad"
    common = dict(encoder="compact", device="cpu", dino_repository=None, weights=str(weights), checkpoint=str(checkpoint))
    commission(argparse.Namespace(**common, manifest=str(manifest), config=str(config)))
    initial = SafeCAD.load(checkpoint)
    rows = [{"image": "image_0.png", "episode": i, "domain": "camera"} for i in range(16)]
    replay = tmp_path/"stream.json"
    replay.write_text(json.dumps(rows))
    output = tmp_path/"online"
    stream(argparse.Namespace(**common, manifest=str(replay), output=str(output), frozen=False, maps=True))
    predictions = [json.loads(row) for row in (output/"predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 16
    assert np.load(output/predictions[0]["anomaly_map"]).shape == (80, 120)
    adapted = SafeCAD.load(output/"final.safecad")
    assert adapted.domains["camera"].completed_window == 3
    assert adapted.domains["camera"].threshold == initial.domains["camera"].threshold
    frozen = tmp_path/"frozen"
    stream(argparse.Namespace(**common, manifest=str(replay), output=str(frozen), frozen=True, maps=False))
    assert not (frozen/"final.safecad").exists()
    assert (frozen/"events.jsonl").read_text() == ""
    assert SafeCAD.load(checkpoint).arrival == 0

    # Exercise the documented train/test commands with retrospective labels.
    from safecad.cli import main
    train_output = tmp_path/"trained"
    main(["train", "--commission-manifest", str(manifest), "--adapt-manifest", str(replay),
          "--output", str(train_output), "--encoder", "compact", "--weights", str(weights), "--config", str(config)])
    assert (train_output/"model.safecad").is_file()
    evaluation = tmp_path/"evaluation.json"
    evaluation.write_text(json.dumps([dict(image="image_0.png", episode=16, domain="camera", label=0),
                                      dict(image="image_1.png", episode=17, domain="camera", label=1)]))
    main(["test", "--manifest", str(evaluation), "--checkpoint", str(train_output/"model.safecad"),
          "--output", str(tmp_path/"tested"), "--encoder", "compact", "--weights", str(weights)])
    metrics = json.loads((tmp_path/"tested/metrics.json").read_text())
    assert metrics["per_domain"]["camera"]["reviewed"] == 2
