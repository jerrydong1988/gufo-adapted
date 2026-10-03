"""Compare matched-token full logits and next-token perplexity."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("tokens", type=Path)
    parser.add_argument("--rmse", type=float, default=0.1)
    parser.add_argument("--max-error", type=float, default=1.0)
    parser.add_argument("--ppl-relative", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--positions", type=Path, help="one-based frontier positions in full token history")
    args = parser.parse_args()
    ids = np.loadtxt(args.tokens, dtype=np.int64, ndmin=1)
    rows = []
    losses = [[], []]
    positions = np.loadtxt(args.positions, dtype=np.int64, ndmin=1) if args.positions else np.arange(1, len(ids) + 1)
    if np.any(positions < 1) or np.any(positions > len(ids)) or len(set(positions)) != len(positions):
        raise ValueError("invalid frontier positions")
    expected = len(positions) * 262144 * 4
    if args.reference.stat().st_size != expected or args.candidate.stat().st_size != expected:
        raise ValueError("incomplete logit file or mismatched token count")
    a = np.memmap(args.reference, dtype="<f4", mode="r", shape=(len(positions), 262144))
    b = np.memmap(args.candidate, dtype="<f4", mode="r", shape=a.shape)
    for i, position in enumerate(positions):
        left, right = np.asarray(a[i], dtype=np.float64), np.asarray(b[i], dtype=np.float64)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"non-finite logits at {i}")
        difference = left - right
        rows.append(dict(position=int(position), rmse=float(np.sqrt(np.mean(difference ** 2))),
                         max_error=float(np.max(np.abs(difference))),
                         reference_top=int(np.argmax(left)), candidate_top=int(np.argmax(right))))
        if position < len(ids):
            for index, row in enumerate((left, right)):
                maximum = np.max(row)
                losses[index].append(float(maximum + np.log(np.exp(row - maximum).sum()) - row[ids[position]]))
    ppl = [float(np.exp(np.mean(values))) for values in losses]
    relative = abs(ppl[1] / ppl[0] - 1)
    summary = dict(rows=len(rows), max_rmse=max(row["rmse"] for row in rows),
                   max_error=max(row["max_error"] for row in rows),
                   top1_agreement=sum(row["reference_top"] == row["candidate_top"] for row in rows) / len(rows),
                   reference_ppl=ppl[0], candidate_ppl=ppl[1], ppl_relative=relative,
                   limits=dict(rmse=args.rmse, max_error=args.max_error, ppl_relative=args.ppl_relative))
    passed = summary["max_rmse"] <= args.rmse and summary["max_error"] <= args.max_error and relative <= args.ppl_relative
    print(json.dumps(dict(passed=passed, **summary), indent=2))
    if args.output:
        args.output.write_text(json.dumps(dict(passed=passed, summary=summary, rows=rows), indent=2) + "\n", encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
