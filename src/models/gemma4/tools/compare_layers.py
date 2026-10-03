"""Compare all named FP32 layer fixtures from independent/native executions."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rmse", type=float, required=True)
    parser.add_argument("--max-error", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"limits": {"rmse": args.rmse, "max_error": args.max_error}, "layers": {}, "passed": True}
    names = sorted(path.name for path in args.candidate.glob("*.bin"))
    if not names: raise ValueError("no candidate layer fixtures")
    for name in names:
        a = np.fromfile(args.reference / name, dtype="<f4").astype(np.float64)
        b = np.fromfile(args.candidate / name, dtype="<f4").astype(np.float64)
        if a.shape != b.shape or not a.size: raise ValueError(f"layer shape differs: {name}")
        finite = np.isfinite(a).all() and np.isfinite(b).all()
        rmse = float(np.sqrt(np.mean((a - b) ** 2))) if finite else None
        maximum = float(np.max(np.abs(a - b))) if finite else None
        passed = bool(finite and rmse <= args.rmse and maximum <= args.max_error)
        report["layers"][name] = {"elements": int(a.size), "rmse": rmse, "max_error": maximum, "passed": passed}
        report["passed"] &= passed
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "layers": len(names), "max_rmse": max((x["rmse"] or 0) for x in report["layers"].values()), "max_error": max((x["max_error"] or 0) for x in report["layers"].values())}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
