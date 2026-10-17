#!/usr/bin/env python
"""Plot mean best_path ``overall.score`` vs generation for planner vs rule_based.

Score direction: ``overall.score`` is ``ScoreBreakdown.total`` (MAPE + SAScore
penalty + minimal-edit penalty) and the beam search ranks ascending, so
LOWER IS BETTER. The y-axis is labelled accordingly.

Variable-length policy (carry-forward): best paths have different lengths
(a run may finish with a best molecule found at generation 2 of 8). For each
run, the score at generation g is the score of the last best_path step with
generation <= g; after the path ends its final score is carried forward up to
the maximum generation observed in that mode (the best molecule is unchanged,
so the achieved score is unchanged). Mean and +/-1 SD (population, ddof=0)
are computed per generation over all runs of a mode; ``n`` per generation is
written to the summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OKABE_ITO = {"planner": "#0072B2", "rule_based": "#E69F00"}
LABELS = {"planner": "Learned planner", "rule_based": "Rule-based modifier"}


def load_runs(log_dir: Path) -> dict[str, list[list[tuple[int, float]]]]:
    runs: dict[str, list] = {"planner": [], "rule_based": []}
    for p in sorted(Path(log_dir).glob("*.json")):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        mode = rec.get("mode")
        steps = [
            (int(s["generation"]), float(s["overall"]["score"]))
            for s in rec.get("best_path") or []
            if s.get("overall", {}).get("score") is not None
        ]
        if mode in runs and steps:
            runs[mode].append(sorted(steps))
    return runs


def carry_forward(steps: list[tuple[int, float]], max_gen: int) -> np.ndarray:
    out = np.full(max_gen + 1, np.nan)
    cur = np.nan
    lookup = dict(steps)
    for g in range(max_gen + 1):
        if g in lookup:
            cur = lookup[g]
        out[g] = cur
    return out


def summarize(runs: list[list[tuple[int, float]]]):
    if not runs:
        return np.array([]), np.array([]), np.array([]), np.array([])
    max_gen = max(g for r in runs for g, _ in r)
    mat = np.vstack([carry_forward(r, max_gen) for r in runs])
    n = np.sum(~np.isnan(mat), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(mat, axis=0)
        sd = np.nanstd(mat, axis=0)
    return np.arange(max_gen + 1), mean, sd, n


def plot(summaries: dict, out_dir: Path, stem: str = "best_path_score_vs_generation") -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, (gen, mean, sd, n) in summaries.items():
        if len(gen) == 0:
            continue
        c = OKABE_ITO[mode]
        ax.plot(gen, mean, color=c, marker="o", lw=2, label=f"{LABELS[mode]} (n={int(n[0])} runs)")
        ax.fill_between(gen, mean - sd, mean + sd, color=c, alpha=0.2, label=f"{LABELS[mode]} ±1 SD")
    ax.set_xlabel("Generation (beam-search expansion round)")
    ax.set_ylabel("Best-path overall score (lower is better)")
    ax.set_title("Best-path score vs generation (carry-forward after path end)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    paths = []
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 300})):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, **kw)
        paths.append(p)
    plt.close(fig)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", type=Path,
                    default=Path(os.environ.get("IGNITE_RUN_LOG_DIR") or ROOT / "logs" / "benchmark"))
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    args = ap.parse_args(argv)
    runs = load_runs(args.log_dir)
    if not any(runs.values()):
        print(f"no valid run logs in {args.log_dir}")
        return 1
    summaries = {m: summarize(r) for m, r in runs.items()}
    paths = plot(summaries, args.out_dir)
    csv_path = args.out_dir / "best_path_score_vs_generation.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "generation", "mean_score", "sd_score", "n_runs"])
        for m, (gen, mean, sd, n) in summaries.items():
            for row in zip(gen, mean, sd, n):
                w.writerow([m, int(row[0]), f"{row[1]:.6g}", f"{row[2]:.6g}", int(row[3])])
    for m, r in runs.items():
        print(f"{m}: {len(r)} runs")
    for p in [*paths, csv_path]:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
