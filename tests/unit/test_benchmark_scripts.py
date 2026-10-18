"""Benchmark runner/analysis on small synthetic logs (no search is executed)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rb = _load("run_benchmark")
ab = _load("analyze_best_path")


def _log(path, mode, scores):
    path.write_text(json.dumps({
        "mode": mode,
        "best_path": [{"step": i, "generation": i, "overall": {"score": s}} for i, s in enumerate(scores)],
    }))


def test_config_has_50_valid_combos():
    from backend.app.domain.models import BeamParameters, TargetProperties

    cfg = rb.load_config(ROOT / "configs" / "benchmark_runs.json")
    assert len(cfg["combos"]) == 50
    assert len({c["id"] for c in cfg["combos"]}) == 50
    assert set(cfg["modes"]) == {"planner", "rule_based"}
    for c in cfg["combos"]:
        TargetProperties(**c["targets"])
        BeamParameters(**{**cfg["fixed_beam"], **c["beam"]})
        assert c["beam"]["depth"] <= 8


def test_resume_skips_only_valid_logs(tmp_path):
    cfg = {"combos": [{"id": "a"}, {"id": "b"}]}
    _log(rb.log_path(tmp_path, "a", "planner"), "planner", [1.0, 0.5])
    rb.log_path(tmp_path, "a", "rule_based").write_text("{broken")
    rb.log_path(tmp_path, "b", "planner").write_text(json.dumps({"best_path": []}))
    todo, skipped = rb.plan(cfg, tmp_path)
    assert [(c["id"], m) for c, m in skipped] == [("a", "planner")]
    assert len(todo) == 3


def test_runner_records_failures(tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps({"combos": [{"id": "x", "targets": {}, "beam": {}}]}))

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(rb, "run_one", boom)
    rc = rb.main(["--config", str(cfg_path), "--log-dir", str(tmp_path / "logs")])
    assert rc == 1
    lines = (tmp_path / "logs" / "failures.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert "synthetic failure" in json.loads(lines[0])["error"]


def test_carry_forward_and_summary():
    runs = [[(0, 1.0), (1, 0.6)], [(0, 0.8), (1, 0.4), (2, 0.2)]]
    gen, mean, sd, n = ab.summarize(runs)
    assert list(gen) == [0, 1, 2]
    np.testing.assert_allclose(mean, [0.9, 0.5, 0.4])  # run 1 carries 0.6 to gen 2
    np.testing.assert_allclose(sd, [0.1, 0.1, 0.2])
    assert list(n) == [2, 2, 2]


def test_analysis_writes_figures(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    _log(logs / "a__planner.json", "planner", [1.0, 0.7, 0.5])
    _log(logs / "b__planner.json", "planner", [0.9, 0.6])
    _log(logs / "a__rule_based.json", "rule_based", [1.0, 0.9])
    (logs / "junk.json").write_text("not json")
    out = tmp_path / "fig"
    assert ab.main(["--log-dir", str(logs), "--out-dir", str(out)]) == 0
    for ext in ("pdf", "svg", "png", "csv"):
        assert (out / f"best_path_score_vs_generation.{ext}").stat().st_size > 0
    from PIL import Image

    dpi = Image.open(out / "best_path_score_vs_generation.png").info.get("dpi")
    assert dpi and round(dpi[0]) == 300


def test_analysis_no_logs(tmp_path):
    assert ab.main(["--log-dir", str(tmp_path), "--out-dir", str(tmp_path / "o")]) == 1


def test_lower_score_is_better_in_beam_ranking():
    from types import SimpleNamespace
    from backend.app.services.beam_search import BeamSearchController

    mk = lambda t: SimpleNamespace(score=SimpleNamespace(total=t), generation=1, id=str(t))
    kept = BeamSearchController._select_survivors([mk(0.9), mk(0.1), mk(0.5)], 1, explore=False)
    assert kept[0].score.total == 0.1
