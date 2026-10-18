"""Generate a benchmark config (default: 500 combos -> configs/benchmark_runs_500.json).

Same schema and rules as configs/benchmark_runs.json: targets inside the ~10th-90th
percentile band of the seed library, det_velocity/det_pressure rising with density
(Kamlet-Jacobs-like) with small noise, beam depth/width/top_k cycled so all values
co-occur, unique random_seed per combo. Deterministic for a fixed --rng-seed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RNG_SEED = 20261018
DENSITY = (1.45, 1.95)
VELOCITY = (6500.0, 9500.0)
PRESSURE = (17.5, 40.5)
HF = (-130.0, 100.0)
DEPTHS = [3, 4, 5, 6, 8]
WIDTHS = [3, 4, 5, 6, 8]
TOP_KS = [2, 3, 4, 5]
FIXED_BEAM = {
    "mape_threshold": 1.0,
    "enable_literature": True,
    "enable_patent_search": False,
    "novelty_as_gate": False,
    "early_stop_on_mape": False,
}


def _clip(x, lo_hi):
    return min(max(x, lo_hi[0]), lo_hi[1])


def generate(n: int = 500, rng_seed: int = RNG_SEED, seed_start: int = 2000) -> dict:
    rng = random.Random(rng_seed)
    width_id = len(str(n))
    combos = []
    for i in range(n):
        d = round(rng.uniform(*DENSITY), 3)
        f = (d - DENSITY[0]) / (DENSITY[1] - DENSITY[0])
        v = round(_clip(VELOCITY[0] + f * (VELOCITY[1] - VELOCITY[0]) + rng.gauss(0, 150), VELOCITY) / 10) * 10.0
        p = round(_clip(PRESSURE[0] + f * (PRESSURE[1] - PRESSURE[0]) + rng.gauss(0, 1.0), PRESSURE), 1)
        hf = round(rng.uniform(*HF), 1)
        combos.append({
            "id": f"combo_{i + 1:0{width_id}d}",
            "targets": {"density": d, "det_velocity": v, "det_pressure": p, "hf_solid": hf},
            "beam": {
                "depth": DEPTHS[i % len(DEPTHS)],
                "width": WIDTHS[(i // len(DEPTHS)) % len(WIDTHS)],
                "top_k": TOP_KS[(i // (len(DEPTHS) * len(WIDTHS))) % len(TOP_KS)],
                "random_seed": seed_start + i,
            },
        })
    return {
        "description": f"{n} benchmark combos for best_path analysis; each is run in planner and rule_based modes.",
        "rationale": {
            "targets": "Same band as configs/benchmark_runs.json: density 1.45-1.95, det_velocity 6500-9500 and det_pressure 17.5-40.5 rising linearly with density plus Gaussian noise (sd 150 m/s, 1 GPa, clipped), hf_solid uniform -130..+100.",
            "beam": "depth in {3,4,5,6,8}, width in {3,4,5,6,8}, top_k in {2,3,4,5}, cycled with periods 1/5/25 so all 100 combinations co-occur. random_seed unique per combo.",
            "fixed": "early_stop_on_mape=false; patent/online search off (no network).",
            "generator": f"scripts/generate_benchmark_config.py, random.Random({rng_seed}), seeds {seed_start}-{seed_start + n - 1}",
        },
        "fixed_beam": dict(FIXED_BEAM),
        "modes": ["planner", "rule_based"],
        "combos": combos,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--rng-seed", type=int, default=RNG_SEED)
    ap.add_argument("--seed-start", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=ROOT / "configs" / "benchmark_runs_500.json")
    a = ap.parse_args(argv)
    a.out.write_text(json.dumps(generate(a.n, a.rng_seed, a.seed_start), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
