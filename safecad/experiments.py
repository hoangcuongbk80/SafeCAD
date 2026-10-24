"""Repeat the Real-IAD protocol across seeds and contamination levels."""
import argparse
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
from .data import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the SafeCAD Real-IAD benchmark")
    parser.add_argument("--root", default="data/Real-IAD")
    parser.add_argument("--output", default="runs/realiad")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    parser.add_argument("--contamination", type=float, nargs="+", default=[0,0.01,0.05,0.10,0.20])
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--encoder", choices=("dinov2", "compact"), default="dinov2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config")
    parser.add_argument("--weights")
    parser.add_argument("--dino-repository")
    parser.add_argument("--task-free", action="store_true")
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument("--aupro", action="store_true")
    parser.add_argument("--metric-max-side", type=int)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    shared = ["--encoder", args.encoder, "--device", args.device]
    for name in ("weights", "dino_repository"):
        if getattr(args, name):
            shared += ["--"+name.replace("_", "-"), getattr(args, name)]
    if args.task_free:
        shared += ["--task-free"]
    for rho in args.contamination:
        for seed in args.seeds:
            run = output/f"rho_{rho:g}"/f"seed_{seed}"
            metrics = run/"test"/"metrics.json"
            if metrics.is_file():
                print(f"Skipping completed run: {run}", flush=True)
                continue
            manifests = run/"manifests"
            prep = [sys.executable, "-m", "safecad.data", "prepare-realiad", "--root", args.root,
                    "--output", str(manifests), "--seed", str(seed), "--contamination", str(rho)]
            if args.categories:
                prep += ["--categories", *args.categories]
            subprocess.run(prep, check=True)
            train = [sys.executable, "-m", "safecad", "train", "--commission-manifest", str(manifests/"commission.json"),
                     "--adapt-manifest", str(manifests/"adaptation.json"), "--output", str(run/"train"), *shared]
            if args.config:
                train += ["--config", args.config]
            subprocess.run(train, check=True)
            test = [sys.executable, "-m", "safecad", "test", "--manifest", str(manifests/"evaluation.json"),
                    "--checkpoint", str(run/"train"/"model.safecad"), "--output", str(run/"test"), *shared]
            for flag in ("image_only", "aupro"):
                if getattr(args, flag):
                    test += ["--"+flag.replace("_", "-")]
            if args.metric_max_side:
                test += ["--metric-max-side", str(args.metric_max_side)]
            subprocess.run(test, check=True)
    # Average seeds within each domain before averaging domains.
    summary = {}
    for rho in args.contamination:
        reports = [json.loads((output/f"rho_{rho:g}"/f"seed_{seed}"/"test"/"metrics.json").read_text()) for seed in args.seeds]
        domains = sorted({name for report in reports for name in report["per_domain"]})
        names = list(reports[0]["macro"])
        per_domain = {}
        for domain in domains:
            per_domain[domain] = {}
            for name in names:
                values = [report["per_domain"][domain][name] for report in reports if report["per_domain"].get(domain, {}).get(name) is not None]
                per_domain[domain][name] = float(np.mean(values)) if values else None
        macro = {}
        for name in names:
            values = [metrics[name] for metrics in per_domain.values() if metrics[name] is not None]
            macro[name] = float(np.mean(values)) if values else None
        summary[f"{rho:g}"] = dict(seeds=args.seeds, per_domain=per_domain, macro=macro)
    write_json(output/"summary.json", summary)


if __name__ == "__main__":
    main()
