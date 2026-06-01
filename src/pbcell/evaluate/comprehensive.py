"""Comprehensive evaluation: run FID + Oracle in one command on generated samples.

Expects generated samples already exist at --generated_dir (organized by class).
Writes a single JSON report with both metrics.

For Clf Δ (augmentation benefit), use pbcell.evaluate.clf_delta separately.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pbcell.evaluate.fid import compute_fid_for_dirs
from pbcell.evaluate.oracle import evaluate_synthetic, load_oracle


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run FID + Oracle evaluation on one sample set.")
    parser.add_argument("--generated_dir", type=str, required=True,
                        help="Directory with generated images (class subdirs)")
    parser.add_argument("--real_dir", type=str, required=True,
                        help="Directory with real images (for FID)")
    parser.add_argument("--oracle_path", type=str, required=True,
                        help="Path to oracle checkpoint (for class fidelity)")
    parser.add_argument("--name", type=str, default="generated",
                        help="Label for this run in the report")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON (default: <generated_dir>/comprehensive_eval.json)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--skip_oracle", action="store_true")
    args = parser.parse_args(argv)

    generated_dir = Path(args.generated_dir)
    output = Path(args.output) if args.output else generated_dir / "comprehensive_eval.json"

    report = {
        "name": args.name,
        "generated_dir": str(generated_dir),
        "real_dir": args.real_dir,
        "oracle_path": args.oracle_path,
        "timestamp": datetime.now().isoformat(),
    }

    if not args.skip_fid:
        print(f"\n{'='*70}\n[1/2] FID\n{'='*70}")
        fid, n_gen, n_real = compute_fid_for_dirs(generated_dir, args.real_dir, args.device)
        report["fid"] = {"value": fid, "n_generated": n_gen, "n_real": n_real}

    if not args.skip_oracle:
        print(f"\n{'='*70}\n[2/2] Oracle accuracy\n{'='*70}")
        model, classes, class_to_idx = load_oracle(Path(args.oracle_path), args.device)
        oracle_results = evaluate_synthetic(
            model=model,
            synth_dir=generated_dir,
            class_to_idx=class_to_idx,
            classes=classes,
            device=args.device,
            batch_size=args.batch_size,
        )
        report["oracle"] = oracle_results

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*70}\nSUMMARY: {args.name}\n{'='*70}")
    if "fid" in report:
        print(f"  FID:              {report['fid']['value']:.2f}")
    if "oracle" in report and report["oracle"]:
        o = report["oracle"]
        print(f"  Oracle accuracy:  {o['accuracy']:.2%}")
        print(f"  Balanced accuracy: {o['balanced_accuracy']:.2%}")
        print(f"  F1 macro:         {o['f1_macro']:.4f}")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
