import json
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("sklearn")
from safecad.evaluation import ranking_metrics, pixel_metrics, operating_metrics, evaluate_files


def test_perfect_ranking_and_undefined_metrics():
    assert ranking_metrics([0,0,1,1], [0.1,0.2,0.8,0.9]) == dict(auroc=100, ap=100)
    assert ranking_metrics([], []) == dict(auroc=None, ap=None)
    assert ranking_metrics([0,0], [0.1,0.2]) == dict(auroc=None, ap=None)


def test_abstention_is_miss_and_anchor_comparison_is_fixed():
    metrics = operating_metrics([0,0,1,1], [0,0,1,0], [1,0,1,0], [1,1,1,0])
    assert metrics["recall"] == 50
    assert metrics["fpr"] == 0
    assert metrics["fpr_gain"] == 50
    assert metrics["coverage"] == 75
    assert metrics["anomaly_coverage"] == 50


def test_perfect_pixel_metrics_and_aupro():
    mask = np.zeros((8,8), bool)
    mask[1:3,1:3] = True
    mask[4:7,4:7] = True
    result = pixel_metrics([(mask.astype(np.float32), mask)], aupro=True)
    assert result["auroc"] == result["ap"] == result["aupro"] == pytest.approx(100)


def test_aupro_gives_equal_weight_to_large_and_small_regions():
    mask = np.zeros((10,10), bool)
    mask[1,1] = True
    mask[5:8,5:8] = True
    scores = np.zeros_like(mask, dtype=np.float32)
    scores[1,1] = 1
    # The small region is detected before any false positives; the larger region
    # ties with all background pixels. At FPR 0.30 the interpolated PRO is 0.65.
    result = pixel_metrics([(scores, mask)], aupro=True)
    assert result["aupro"] == pytest.approx(57.5)


def test_file_evaluation_aligns_predictions_masks_and_domains(tmp_path):
    mask = np.zeros((12,16), np.uint8)
    mask[3:6,4:7] = 255
    Image.fromarray(mask).save(tmp_path/"mask.png")
    rows = [dict(image="normal.png", domain="camera", episode=0, label=0),
            dict(image="defect.png", domain="camera", episode=1, label=1, mask="mask.png")]
    predictions = []
    for i, row in enumerate(rows):
        score = np.zeros(mask.shape, np.float32) if i == 0 else (mask > 0).astype(np.float32)
        np.save(tmp_path/f"map_{i}.npy", score)
        predictions.append(dict(index=i, image=row["image"], abstained=False, score=float(i),
            threshold=0.5, anchor_score=float(i), anomalous=bool(i), anomaly_map=f"map_{i}.npy"))
    manifest, prediction_path = tmp_path/"evaluation.json", tmp_path/"predictions.jsonl"
    manifest.write_text(json.dumps(rows))
    prediction_path.write_text("\n".join(json.dumps(p) for p in predictions))
    result = evaluate_files(manifest, prediction_path, tmp_path/"metrics.json", aupro=True)
    assert result["macro"]["pixel_ap"] == 100
    assert result["macro"]["recall"] == 100
    predictions.reverse()
    prediction_path.write_text("\n".join(json.dumps(p) for p in predictions))
    with pytest.raises(ValueError, match="order differs"):
        evaluate_files(manifest, prediction_path, tmp_path/"metrics.json")
