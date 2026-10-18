#!/usr/bin/env python
"""Run every combo in configs/benchmark_runs.json in planner and rule_based modes.

Resumable: a (combo, mode) pair whose log ``<log_dir>/<combo_id>__<mode>.json``
already exists and is valid (parseable JSON with a non-empty ``best_path``)
is skipped. Failures are appended to ``<log_dir>/failures.jsonl`` and the run
continues. Log directory: ``--log-dir`` > ``$IGNITE_RUN_LOG_DIR`` > ``logs/benchmark``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ("planner", "rule_based")

SUITES = {
    "50": ROOT / "configs" / "benchmark_runs.json",
    "500": ROOT / "configs" / "benchmark_runs_500.json",
}


def log_path(log_dir: Path, combo_id: str, mode: str) -> Path:
    return log_dir / f"{combo_id}__{mode}.json"


def is_valid_log(path: Path) -> bool:
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(rec, dict) and bool(rec.get("best_path"))


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not cfg.get("combos"):
        raise ValueError(f"{path} has no combos")
    return cfg


def plan(cfg: dict, log_dir: Path, modes=MODES, limit=None):
    """Return (todo, skipped) lists of (combo, mode)."""
    combos = cfg["combos"][:limit] if limit else cfg["combos"]
    todo, skipped = [], []
    for combo in combos:
        for mode in modes:
            (skipped if is_valid_log(log_path(log_dir, combo["id"], mode)) else todo).append((combo, mode))
    return todo, skipped


def run_one(combo: dict, mode: str, fixed: dict, depth_override=None) -> dict:
    os.environ.setdefault("IGNITE_PUBCHEM_ENABLED", "false")
    os.environ.setdefault("IGNITE_OPENALEX_ENABLED", "false")
    from backend.app.domain.models import BeamParameters, ModificationMode, TargetProperties
    from backend.app.main import _build_controller, _clamp_beam

    beam_kw = {**fixed, **combo["beam"]}
    if depth_override:
        beam_kw["depth"] = depth_override
    beam = _clamp_beam(BeamParameters(**beam_kw))
    controller = _build_controller(beam)
    targets = TargetProperties(**combo["targets"])
    controller.run(f"{combo['id']}__{mode}", targets, beam, mode=ModificationMode(mode))
    rec = controller.last_run_log
    if not rec or not rec.get("best_path"):
        raise RuntimeError("run produced no best_path")
    rec["benchmark"] = {"combo_id": combo["id"], "beam": beam.model_dump()}
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None, help="Config path (overrides --suite)")
    ap.add_argument("--suite", choices=sorted(SUITES), default="50", help="Built-in config: 50 (default) or 500")
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--limit", type=int, default=None, help="Only the first N combos (smoke test)")
    ap.add_argument("--depth-override", type=int, default=None, help="Force depth (smoke test)")
    ap.add_argument("--dry-run", action="store_true", help="List pending pairs and exit")
    args = ap.parse_args(argv)

    log_dir = args.log_dir or Path(os.environ.get("IGNITE_RUN_LOG_DIR") or ROOT / "logs" / "benchmark")
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.config or SUITES[args.suite]
    cfg = load_config(config_path)
    print(f"config={config_path}", flush=True)
    todo, skipped = plan(cfg, log_dir, args.modes, args.limit)
    print(f"log_dir={log_dir} pending={len(todo)} skipped_valid={len(skipped)}", flush=True)
    if args.dry_run:
        for combo, mode in todo:
            print(f"  {combo['id']} {mode}")
        return 0

    failures = 0
    for i, (combo, mode) in enumerate(todo, 1):
        t0 = time.time()
        try:
            rec = run_one(combo, mode, cfg.get("fixed_beam", {}), args.depth_override)
            rec["benchmark"]["elapsed_s"] = round(time.time() - t0, 2)
            out = log_path(log_dir, combo["id"], mode)
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
            tmp.replace(out)
            print(f"[{i}/{len(todo)}] OK   {combo['id']} {mode} {time.time()-t0:.1f}s", flush=True)
        except Exception as e:  # record and continue
            failures += 1
            with open(log_dir / "failures.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "combo_id": combo["id"], "mode": mode,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }) + "\n")
            print(f"[{i}/{len(todo)}] FAIL {combo['id']} {mode}: {type(e).__name__}: {e}", flush=True)
    print(f"done: ran={len(todo)} failures={failures} skipped={len(skipped)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
