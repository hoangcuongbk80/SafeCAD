"""Dataset acquisition and leakage-free acquisition manifest preparation."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
from zipfile import ZipFile
import numpy as np
from PIL import Image


SOURCE_POSITIONS = {2, 7, 12, 17, 22, 27}
CONTAMINATION_COUNTS = {0.0: 0, 0.01: 3, 0.05: 13, 0.10: 26, 0.20: 51}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_metadata(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError:
        metadata = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(metadata, dict):
        return metadata.get("train", []) + metadata.get("test", [])
    if isinstance(metadata, list):
        return metadata
    raise ValueError(f"Unsupported metadata structure: {path}")


def physical_sample(path):
    """Camera and timestamp suffixes do not change physical sample identity."""
    stem = Path(path).stem
    match = re.match(r"(.+)_C[1-5](?:_|$)", stem, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot identify physical sample from {path}")
    return match.group(1)


def _relative(path, base):
    return Path(os.path.relpath(Path(path).resolve(), Path(base).resolve())).as_posix()


def _image_path(root, resolution, category, value):
    path = Path(str(value).replace("\\", "/"))
    candidates = [root/f"realiad_{resolution}"/category/path,
                  root/f"realiad_{resolution}"/path, root/path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing Real-IAD file: {value} ({category})")


def _metadata_directory(root, explicit):
    if explicit:
        directory = Path(explicit).resolve()
        if not list(directory.glob("*.json")):
            raise FileNotFoundError(f"No category JSON files in {directory}")
        return directory
    for relative in ("realiad_jsons", "realiad_jsons/realiad_jsons", "realiad_jsons_sv", "realiad_jsons/realiad_jsons_sv"):
        directory = root/relative
        if list(directory.glob("*.json")):
            return directory
    raise FileNotFoundError("Extract realiad_jsons.zip under the dataset root, or pass --metadata-dir")


def prepare_realiad(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    metadata_dir = _metadata_directory(root, args.metadata_dir)
    names = sorted(p.stem for p in metadata_dir.glob("*.json"))
    selected = [name for index, name in enumerate(names, 1)
                if args.partition == "all" or (index in SOURCE_POSITIONS) == (args.partition == "source")]
    if args.categories:
        absent = set(args.categories)-set(names)
        if absent:
            raise ValueError(f"Categories missing from metadata: {sorted(absent)}")
        selected = [name for name in selected if name in args.categories]
    if not selected:
        raise ValueError("No categories selected")
    rng = np.random.default_rng(args.seed)
    defective_count = CONTAMINATION_COUNTS.get(args.contamination)
    if args.stream_size != 256 or defective_count is None:
        defective_count = int(np.floor(args.contamination*args.stream_size+0.5))
    if not 0 <= args.contamination <= 1:
        raise ValueError("contamination must lie in [0, 1]")
    commissioning, adaptation, labels, evaluation, summary = [], [], [], [], []
    episode = 0
    for name in selected:
        samples = {}
        for row in read_metadata(metadata_dir/f"{name}.json"):
            image_value = row.get("image_path", row.get("filename"))
            if not image_value:
                raise ValueError(f"Missing image_path/filename in {name}")
            if not re.search(rf"_{re.escape(args.view)}(?:_|\.)", str(image_value), re.IGNORECASE):
                continue
            label = int(row["label"]) if "label" in row else int(row.get("anomaly_class") != "OK")
            if label not in (0, 1):
                raise ValueError("Real-IAD labels must be zero or one")
            image = _image_path(root, args.resolution, name, image_value)
            mask_value = row.get("mask_path", row.get("maskname"))
            mask = _image_path(root, args.resolution, name, mask_value) if mask_value else None
            if label:
                if mask is None:
                    raise ValueError(f"Defective image has no mask: {image}")
                with Image.open(mask) as im:
                    if not np.any(np.asarray(im.convert("L")) > 0):
                        continue
            sample_id = physical_sample(image)
            record = dict(image=_relative(image, output), domain=name, sample_id=sample_id, label=label,
                          mask=_relative(mask, output) if mask else None)
            if sample_id in samples and samples[sample_id]["label"] != label:
                raise ValueError(f"Conflicting labels for physical sample {sample_id}")
            if sample_id not in samples or record["image"] < samples[sample_id]["image"]:
                samples[sample_id] = record
        normal = [samples[k] for k in sorted(samples) if not samples[k]["label"]]
        abnormal = [samples[k] for k in sorted(samples) if samples[k]["label"]]
        normal = [normal[i] for i in rng.permutation(len(normal))]
        abnormal = [abnormal[i] for i in rng.permutation(len(abnormal))]
        n_normal_stream = args.stream_size-defective_count
        required_normal = args.initialization+args.calibration+n_normal_stream+args.test_normal
        required_abnormal = defective_count+args.test_anomalous
        if len(normal) < required_normal or len(abnormal) < required_abnormal:
            raise ValueError(f"{name}: need {required_normal} normal/{required_abnormal} defective physical samples; "
                             f"found {len(normal)}/{len(abnormal)} with view {args.view}")
        a, b = args.initialization, args.initialization+args.calibration
        commissioning.append(dict(id=name, initialization=[r["image"] for r in normal[:a]],
                                  calibration=[r["image"] for r in normal[a:b]]))
        # Reserve the test cohort before varying the stream contamination level.
        stream = normal[b+args.test_normal:b+args.test_normal+n_normal_stream] + abnormal[args.test_anomalous:required_abnormal]
        stream = [stream[i] for i in rng.permutation(len(stream))]
        for row in stream:
            adaptation.append({k: row[k] for k in ("image", "domain", "sample_id")} | {"episode": episode})
            labels.append(row | {"episode": episode})
            episode += 1
        test_rows = normal[b:b+args.test_normal] + abnormal[:args.test_anomalous]
        for row in test_rows:
            evaluation.append(row | {"episode": episode})
            episode += 1
        summary.append(dict(domain=name, initialization=a, calibration=args.calibration,
                            adaptation=len(stream), adaptation_anomalous=defective_count,
                            test_normal=args.test_normal, test_anomalous=args.test_anomalous))
    # Acquisition identifiers remain monotone when evaluation continues adaptation.
    for i, (observation, annotation) in enumerate(zip(adaptation, labels)):
        observation["episode"] = annotation["episode"] = i
    for i, observation in enumerate(evaluation, start=len(adaptation)):
        observation["episode"] = i
    write_json(output/"commission.json", {"domains": commissioning})
    write_json(output/"adaptation.json", adaptation)
    write_json(output/"adaptation_labels.json", labels)
    write_json(output/"evaluation.json", evaluation)
    write_json(output/"protocol.json", dict(dataset="Real-IAD", seed=args.seed, partition=args.partition,
        view=args.view, nominal_contamination=args.contamination,
        realized_contamination=defective_count/args.stream_size, domains=summary))
    print(f"Prepared {len(selected)} domains: {output}")


def safe_extract(archive, destination, category=None, resolution=None):
    destination = Path(destination).resolve()
    with ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if resolution and parts[0] == f"realiad_{resolution}":
                parts = parts[1:]
            if category and parts[0] != category:
                parts.insert(0, category)
            if any(p in ("", ".", "..") or ":" in p for p in parts):
                raise ValueError(f"Unsafe archive member: {info.filename}")
            target = destination.joinpath(*parts).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive member escapes destination: {info.filename}")
            if target.is_file() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def download_realiad(args):
    from huggingface_hub import snapshot_download
    root = Path(args.root).resolve()
    patterns = ["realiad_jsons.zip"]
    patterns += [f"realiad_{args.resolution}/{name}.zip" for name in args.categories] if args.categories else [f"realiad_{args.resolution}/*.zip"]
    archives = Path(snapshot_download("Real-IAD/Real-IAD", repo_type="dataset", local_dir=root/"archives",
                                     allow_patterns=patterns))
    metadata_archive = archives/"realiad_jsons.zip"
    if not metadata_archive.is_file():
        raise FileNotFoundError("Dataset download did not contain realiad_jsons.zip")
    safe_extract(metadata_archive, root)
    image_archives = sorted((archives/f"realiad_{args.resolution}").glob("*.zip"))
    if args.categories:
        image_archives = [p for p in image_archives if p.stem in args.categories]
    if not image_archives or args.categories and len(image_archives) != len(set(args.categories)):
        raise FileNotFoundError("One or more requested category archives were not downloaded")
    for archive in image_archives:
        print(f"Extracting {archive.name}")
        safe_extract(archive, root/f"realiad_{args.resolution}", archive.stem, args.resolution)
    print(f"Real-IAD is ready at {root}")


def prepare_coal(args):
    index, output = Path(args.index).resolve(), Path(args.output).resolve()
    with index.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"image", "domain", "session", "timestamp", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV requires columns: {sorted(required)}")
    if args.camera:
        rows = [r for r in rows if r["domain"] == args.camera]
    if not rows:
        raise ValueError("No frames selected")
    for row in rows:
        row["timestamp"] = float(row["timestamp"])
        if not np.isfinite(row["timestamp"]):
            raise ValueError("Acquisition timestamps must be finite")
        row["image"] = _relative(index.parent/row["image"], output)
        if not (output/row["image"]).is_file():
            raise FileNotFoundError(output/row["image"])
        if row.get("mask"):
            row["mask"] = _relative(index.parent/row["mask"], output)
        row["label"] = int(row["label"]) if row.get("label") not in (None, "", "-1") else None
        if row["label"] not in (None, 0, 1):
            raise ValueError("label must be 0, 1, or empty for unreviewed frames")
        if row["split"] not in ("initialization", "calibration", "adaptation", "evaluation"):
            raise ValueError(f"Invalid split: {row['split']}")
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("A frame appears more than once in the acquisition index")
    sessions = {}
    events = {}
    for row in rows:
        key = row["domain"], row["session"]
        split = "commissioning" if row["split"] in ("initialization", "calibration") else row["split"]
        if key in sessions and sessions[key] != split:
            raise ValueError(f"Session crosses acquisition splits: {key}")
        sessions[key] = split
        if row.get("event_id"):
            event = row["domain"], row["event_id"]
            if event in events and events[event] != split:
                raise ValueError(f"Event crosses acquisition splits: {event}")
            events[event] = split
    commissioning = []
    for domain in sorted({r["domain"] for r in rows}):
        periods = {phase: [r["timestamp"] for r in rows if r["domain"] == domain
                          and (r["split"] in ("initialization", "calibration") if phase == "commissioning" else r["split"] == phase)]
                   for phase in ("commissioning", "adaptation", "evaluation")}
        present = [periods[phase] for phase in ("commissioning", "adaptation", "evaluation") if periods[phase]]
        if any(max(a) >= min(b) for a,b in zip(present, present[1:])):
            raise ValueError(f"{domain}: commissioning, adaptation and evaluation periods must be chronological and disjoint")
        init = sorted([r for r in rows if r["domain"] == domain and r["split"] == "initialization"], key=lambda r:r["timestamp"])
        cal = sorted([r for r in rows if r["domain"] == domain and r["split"] == "calibration"], key=lambda r:r["timestamp"])
        if not init or len(cal) < args.calibration_min or any(r["label"] != 0 for r in init+cal):
            raise ValueError(f"{domain}: supply verified normal initialization and >= {args.calibration_min} calibration frames")
        if any(b["timestamp"]-a["timestamp"] < args.calibration_spacing for a,b in zip(cal, cal[1:])):
            raise ValueError(f"{domain}: calibration frames must be >= {args.calibration_spacing} seconds apart")
        commissioning.append(dict(id=domain, initialization=[r["image"] for r in init], calibration=[r["image"] for r in cal]))
    write_json(output/"commission.json", {"domains": commissioning})
    episode = 0
    for split in ("adaptation", "evaluation"):
        groups = {}
        for r in rows:
            if r["split"] == split:
                key = r["domain"], r["session"], int(np.floor(r["timestamp"]/args.episode_seconds))
                groups.setdefault(key, []).append(r)
        ordered = sorted(groups, key=lambda k: (k[0], min(r["timestamp"] for r in groups[k]), k[1], k[2]))
        manifest, annotations = [], []
        for key in ordered:
            for row in sorted(groups[key], key=lambda r:r["timestamp"]):
                clean = {k:row[k] for k in ("image", "domain", "session", "timestamp")}
                clean["episode"] = episode
                annotations.append(row | {"episode": episode})
                manifest.append(clean if split == "adaptation" else row | {"episode": episode})
            episode += 1
        write_json(output/f"{split}.json", manifest)
        if split == "adaptation":
            write_json(output/"adaptation_labels.json", annotations)
    write_json(output/"protocol.json", dict(dataset="CoalConveyor-AD", episode_seconds=args.episode_seconds,
                                           cameras=[d["id"] for d in commissioning]))
    print(f"Prepared acquisition manifests: {output}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="SafeCAD dataset preparation")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download-realiad")
    download.add_argument("--root", default="data/Real-IAD")
    download.add_argument("--resolution", choices=("256", "512", "1024", "raw"), default="1024")
    download.add_argument("--categories", nargs="+")
    download.set_defaults(func=download_realiad)
    prepare = sub.add_parser("prepare-realiad")
    prepare.add_argument("--root", default="data/Real-IAD")
    prepare.add_argument("--metadata-dir")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--resolution", choices=("256", "512", "1024", "raw"), default="1024")
    prepare.add_argument("--categories", nargs="+")
    prepare.add_argument("--partition", choices=("source", "target", "all"), default="target")
    prepare.add_argument("--view", choices=("C1", "C2", "C3", "C4", "C5"), default="C1")
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--contamination", type=float, default=0.05)
    for name, default in (("initialization",16), ("calibration",20), ("stream-size",256), ("test-normal",128), ("test-anomalous",64)):
        prepare.add_argument(f"--{name}", type=int, default=default)
    prepare.set_defaults(func=prepare_realiad)
    coal = sub.add_parser("prepare-coal")
    coal.add_argument("--index", required=True)
    coal.add_argument("--output", required=True)
    coal.add_argument("--camera")
    coal.add_argument("--episode-seconds", type=float, default=10)
    coal.add_argument("--calibration-min", type=int, default=20)
    coal.add_argument("--calibration-spacing", type=float, default=60)
    coal.set_defaults(func=prepare_coal)
    args = parser.parse_args(argv)
    for name in ("initialization", "calibration", "stream_size", "test_normal", "test_anomalous", "episode_seconds", "calibration_min"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
