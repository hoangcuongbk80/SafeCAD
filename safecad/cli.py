"""Ordered JSON manifests keep data partitioning and acquisition metadata explicit."""
import argparse
import hashlib
import json
import shutil
from argparse import Namespace
from pathlib import Path
import numpy as np
from .config import Config
from .core import SafeCAD
from .features import top_mean


def _read(path):
    path = Path(path).resolve()
    return path.parent, json.loads(path.read_text(encoding="utf-8"))


def _resolve(base, value):
    return (base / value).resolve()


def _processor(args, config):
    from .vision import FrozenEncoder, ImageProcessor
    return ImageProcessor(FrozenEncoder(args.encoder, args.device, args.dino_repository, args.weights), config.trim_fraction)


def _spec(args):
    digest = None
    if args.weights:
        sha = hashlib.sha256()
        with open(args.weights, "rb") as f:
            for block in iter(lambda: f.read(1024*1024), b""):
                sha.update(block)
        digest = sha.hexdigest()
    return dict(encoder=args.encoder, weights_sha256=digest, preprocessing_version=1)


def commission(args):
    base, manifest = _read(args.manifest)
    config_values = json.loads(Path(args.config).read_text()) if args.config else {}
    config = Config.compact(**config_values) if args.encoder == "compact" else Config(**config_values)
    model = SafeCAD(config)
    model.representation = _spec(args)
    processor = _processor(args, config)
    for item in manifest["domains"]:
        init_paths = [_resolve(base, p) for p in item["initialization"]]
        cal_paths = [_resolve(base, p) for p in item["calibration"]]
        if set(init_paths) & set(cal_paths):
            raise ValueError("initialization and calibration paths overlap")
        if len(set(init_paths)) != len(init_paths) or len(set(cal_paths)) != len(cal_paths):
            raise ValueError("commissioning subsets must contain distinct image paths")
        initialization = [processor.extract(p, consistency=False)[0] for p in init_paths]
        calibration = [processor.extract(p, consistency=False)[0] for p in cal_paths]
        model.commission(item["id"], initialization, calibration)
        print(f"Commissioned {item['id']}: threshold={model.domains[item['id']].threshold:.6f}")
    model.save(args.checkpoint)


def stream(args):
    base, records = _read(args.manifest)
    if not isinstance(records, list):
        raise ValueError("stream manifest must be an ordered list")
    for i, row in enumerate(records):
        if type(row.get("episode")) is not int or row["episode"] < 0:
            raise ValueError("each stream row needs a nonnegative integer episode ID")
        if i and row["episode"] < records[i-1]["episode"]:
            raise ValueError("episode IDs must increase between contiguous episodes")
        if i and row["episode"] == records[i-1]["episode"] and row.get("domain") != records[i-1].get("domain"):
            raise ValueError("supplied domain must remain fixed within an episode")
    if getattr(args, "task_free", False):
        records = [{k:v for k,v in row.items() if k != "domain"} for row in records]
    model = SafeCAD.load(args.checkpoint)
    if getattr(model, "representation", None) != _spec(args):
        raise ValueError("encoder/weights differ from the commissioned representation")
    if model.open_episode is not None:
        raise ValueError("CLI replay requires a checkpoint at a completed episode")
    processor = _processor(args, model.config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    events = open(output/"events.jsonl", "w", encoding="utf-8")
    model.event_sink = lambda event: events.write(json.dumps(event)+"\n")
    print(f"{'Frozen evaluation' if args.frozen else 'Causal adaptation'}: {len(records)} observations", flush=True)
    last_reported_boundary = None
    try:
        with open(output/"predictions.jsonl", "w", encoding="utf-8") as f:
            for i, row in enumerate(records):
                episode = row["episode"]
                end = i+1 == len(records) or records[i+1]["episode"] != episode
                features, geometry = processor.extract(_resolve(base, row["image"]),
                    consistency=not args.frozen and model.needs_consistency())
                if args.frozen:
                    # Preserve first-observation routing even when adaptation is disabled.
                    if i == 0 or records[i-1]["episode"] != episode:
                        if i and episode <= records[i-1]["episode"]:
                            raise ValueError("episode IDs must increase")
                        selected, near = model.route(features.descriptor(model.config.trim_fraction)) if row.get("domain") is None else (
                            (row["domain"], row["domain"]) if row["domain"] in model.domains else (None, None))
                    prediction = model._predict(features, selected, near)
                else:
                    prediction = model.observe(features, episode, row.get("domain"), end_episode=end)
                    if model.last_boundary is not None and model.last_boundary != last_reported_boundary:
                        last_reported_boundary = model.last_boundary.copy()
                        info = model.last_boundary
                        print(f"{info['domain']} window={info['window']+1} pending={info['pending']} active={info['active']}", flush=True)
                result = dict(index=i, image=row["image"], episode=episode, domain=prediction.domain,
                    nearest_domain=prediction.nearest_domain, abstained=prediction.abstained,
                    calibrated=prediction.calibrated, score=prediction.score,
                    threshold=prediction.threshold, anomalous=prediction.anomalous,
                    anchor_score=top_mean(prediction.anchor_scores, model.config.top_fraction)
                        if prediction.anchor_scores is not None else None)
                if prediction.patch_scores is not None and args.maps:
                    name = f"map_{i:08d}.npy"
                    np.save(output/name, processor.anomaly_map(prediction.patch_scores, geometry))
                    result["anomaly_map"] = name
                f.write(json.dumps(result)+"\n")
        if not args.frozen:
            model.save(output/"final.safecad")
    finally:
        events.close()


def train(args):
    """Commission immutable references, then adapt QCM on the unlabeled stream."""
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    initial_path = output/"initial.safecad"
    commissioning_args = Namespace(**vars(args))
    commissioning_args.manifest = args.commission_manifest
    commissioning_args.checkpoint = str(initial_path)
    commission(commissioning_args)
    if args.adapt_manifest:
        adaptation_args = Namespace(**vars(args))
        adaptation_args.manifest = args.adapt_manifest
        adaptation_args.checkpoint = str(initial_path)
        adaptation_args.output = str(output/"adaptation")
        adaptation_args.frozen = False
        adaptation_args.maps = False
        stream(adaptation_args)
        source = output/"adaptation"/"final.safecad"
    else:
        source = initial_path
    shutil.copyfile(source, output/"model.safecad")
    from .data import write_json
    write_json(output/"train_config.json", {k:v for k,v in vars(args).items() if k != "func"})
    print(f"Saved SafeCAD checkpoint: {output/'model.safecad'}")


def test(args):
    replay = Namespace(**vars(args))
    replay.frozen = not args.online
    replay.maps = not args.image_only
    stream(replay)
    from .evaluation import evaluate_files
    evaluate_files(args.manifest, Path(args.output)/"predictions.jsonl", Path(args.output)/"metrics.json",
                   image_only=args.image_only, aupro=args.aupro, max_side=args.metric_max_side)


def main(argv=None):
    parser = argparse.ArgumentParser(description="SafeCAD training, adaptation and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "test", "commission", "stream"):
        p = sub.add_parser(command)
        if command == "train":
            p.add_argument("--commission-manifest", required=True)
            p.add_argument("--adapt-manifest")
        else:
            p.add_argument("--manifest", required=True)
            p.add_argument("--checkpoint", required=True)
        p.add_argument("--encoder", choices=("dinov2", "compact"), default="dinov2")
        p.add_argument("--device", default="cpu")
        p.add_argument("--dino-repository", help="optional local official DINOv2 checkout")
        p.add_argument("--weights", help="optional local model state_dict")
        if command in ("commission", "train"):
            p.add_argument("--config", help="JSON Config field overrides")
        if command in ("train", "test", "stream"):
            p.add_argument("--output", required=True)
            p.add_argument("--task-free", action="store_true", help="route episodes without supplied domain IDs")
        if command == "stream":
            p.add_argument("--frozen", action="store_true")
            p.add_argument("--maps", action="store_true")
        if command == "test":
            p.add_argument("--online", action="store_true", help="continue causal adaptation during evaluation")
            p.add_argument("--image-only", action="store_true", help="skip anomaly maps and pixel metrics")
            p.add_argument("--aupro", action="store_true", help="compute region overlap up to FPR 0.30")
            p.add_argument("--metric-max-side", type=int, help="resize maps/masks for reduced-resolution pixel evaluation")
        p.set_defaults(func={"commission":commission, "stream":stream, "train":train, "test":test}[command])
    args = parser.parse_args(argv)
    if getattr(args, "metric_max_side", None) is not None and args.metric_max_side < 1:
        parser.error("--metric-max-side must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
