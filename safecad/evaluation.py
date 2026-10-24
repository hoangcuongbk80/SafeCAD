"""Retrospective detection metrics. Ground truth never enters detector updates."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .data import write_json


def ranking_metrics(labels, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, s = np.asarray(labels, dtype=np.uint8), np.asarray(scores, dtype=np.float32)
    if not len(y):
        return dict(auroc=None, ap=None)
    return dict(auroc=float(100*roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None,
                ap=float(100*average_precision_score(y, s)) if y.any() else None)


def operating_metrics(labels, decisions, anchor_decisions, covered):
    y = np.asarray(labels, dtype=bool)
    positive = np.asarray(decisions, dtype=bool)
    anchors = np.asarray(anchor_decisions, dtype=bool)
    covered = np.asarray(covered, dtype=bool)
    def rate(values, mask):
        return float(100*np.mean(values[mask])) if mask.any() else None
    recall = rate(positive, y)
    fpr = rate(positive, ~y)
    anchor_recall = rate(anchors, y)
    anchor_fpr = rate(anchors, ~y)
    return dict(recall=recall, fpr=fpr, anchor_recall=anchor_recall, anchor_fpr=anchor_fpr,
                recall_loss=anchor_recall-recall if recall is not None else None,
                fpr_gain=anchor_fpr-fpr if fpr is not None else None,
                coverage=float(100*covered.mean()) if len(y) else None,
                anomaly_coverage=rate(covered, y), normal_coverage=rate(covered, ~y))


def pixel_metrics(pairs, aupro=False):
    """Exact pixel ranking; weighted ROC integrates equal-weight region overlap."""
    if not pairs:
        return dict(auroc=None, ap=None, aupro=None)
    labels = np.concatenate([mask.reshape(-1) for _, mask in pairs])
    scores = np.concatenate([score.reshape(-1) for score, _ in pairs])
    metrics = ranking_metrics(labels, scores)
    metrics["aupro"] = None
    if aupro and labels.any() and (~labels).any():
        from scipy.ndimage import label as component_label
        from scipy.integrate import trapezoid
        from sklearn.metrics import roc_curve
        weights = []
        for _, mask in pairs:
            components, n = component_label(mask, structure=np.ones((3, 3)))
            area = np.bincount(components.ravel(), minlength=n+1)
            weight = np.ones(n+1, dtype=np.float32)
            weight[1:] = 1/area[1:]
            weights.append(weight[components].reshape(-1))
        fpr, pro, _ = roc_curve(labels, scores, sample_weight=np.concatenate(weights), drop_intermediate=False)
        # Keep all tied-score operating points and interpolate at the 0.30 limit.
        keep = fpr <= 0.30
        xx, yy = fpr[keep], pro[keep]
        if xx[-1] < 0.30:
            xx = np.append(xx, 0.30)
            yy = np.append(yy, np.interp(0.30, fpr, pro))
        metrics["aupro"] = float(100*trapezoid(yy, xx)/0.30)
    return metrics


def _pixel_pair(row, prediction, base, prediction_dir, max_side):
    if not prediction.get("anomaly_map") or prediction["abstained"]:
        return None
    score = np.load(prediction_dir/prediction["anomaly_map"], allow_pickle=False)
    if score.ndim != 2 or not np.isfinite(score).all():
        raise ValueError("Anomaly map must be finite [height, width]")
    if row.get("mask"):
        with Image.open(base/row["mask"]) as image:
            image = image.convert("L").resize((score.shape[1], score.shape[0]), Image.Resampling.NEAREST)
            mask = np.asarray(image) > 0
    elif row["label"] == 0:
        mask = np.zeros(score.shape, dtype=bool)
    else:
        return None
    if max_side and max(score.shape) > max_side:
        scale = max_side/max(score.shape)
        size = max(1, round(score.shape[1]*scale)), max(1, round(score.shape[0]*scale))
        score = np.asarray(Image.fromarray(score).resize(size, Image.Resampling.BILINEAR))
        mask = np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST), dtype=bool)
    return score, mask


def evaluate_files(manifest_path, predictions_path, output_path, image_only=False, aupro=False, max_side=None):
    manifest_path, predictions_path = Path(manifest_path).resolve(), Path(predictions_path).resolve()
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    predictions = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(predictions):
        raise ValueError("Prediction and evaluation manifest lengths differ")
    grouped = {}
    for i, (row, prediction) in enumerate(zip(rows, predictions)):
        if prediction["index"] != i or prediction["image"] != row["image"]:
            raise ValueError(f"Prediction order differs from the manifest at row {i}")
        grouped.setdefault(row.get("domain", "default"), []).append((row, prediction))
    results = {}
    for domain, observations in grouped.items():
        reviewed = [(r,p) for r,p in observations if r.get("label") in (0,1)]
        labels = [r["label"] for r,p in reviewed]
        covered = [not p["abstained"] for r,p in reviewed]
        decisions = [p["anomalous"] is True for r,p in reviewed]
        anchor_decisions = [not p["abstained"] and p["anchor_score"] > p["threshold"] for r,p in reviewed]
        rank = ranking_metrics([r["label"] for r,p in reviewed if not p["abstained"]],
                               [p["score"] for r,p in reviewed if not p["abstained"]])
        metrics = {"observations": len(observations), "reviewed": len(reviewed),
                   "image_auroc": rank["auroc"], "image_ap": rank["ap"]}
        metrics.update(operating_metrics(labels, decisions, anchor_decisions, covered))
        pairs = []
        if not image_only:
            for row, prediction in reviewed:
                pair = _pixel_pair(row, prediction, manifest_path.parent, predictions_path.parent, max_side)
                if pair is not None:
                    pairs.append(pair)
        pixel = pixel_metrics(pairs, aupro)
        metrics.update(pixel_auroc=pixel["auroc"], pixel_ap=pixel["ap"], aupro=pixel["aupro"],
                       pixel_annotated_images=len(pairs))
        results[domain] = metrics
        print(f"{domain}: I-AUROC={metrics['image_auroc']} P-AP={metrics['pixel_ap']} "
              f"FPR={metrics['fpr']} Recall={metrics['recall']} Coverage={metrics['coverage']}")
    names = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro", "recall", "fpr", "recall_loss", "fpr_gain", "coverage")
    macro = {}
    for name in names:
        values = [m[name] for m in results.values() if m[name] is not None]
        macro[name] = float(np.mean(values)) if values else None
    report = dict(per_domain=results, macro=macro, metric_scale="percent; gain/loss in percentage points",
                  ranking_population="covered observations", operating_population="all reviewed observations; abstention is a negative decision",
                  metric_max_side=max_side, aupro_fpr_limit=0.30 if aupro else None)
    write_json(output_path, report)
    return report
