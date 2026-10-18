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


gen = _load("generate_benchmark_config")


def test_config_500_valid_unique_in_range():
    from backend.app.domain.models import BeamParameters, TargetProperties

    cfg = rb.load_config(ROOT / "configs" / "benchmark_runs_500.json")
    combos = cfg["combos"]
    assert len(combos) == 500
    assert len({c["id"] for c in combos}) == 500
    assert len({c["beam"]["random_seed"] for c in combos}) == 500
    assert set(cfg["modes"]) == {"planner", "rule_based"}
    assert cfg["fixed_beam"]["early_stop_on_mape"] is False
    assert cfg["fixed_beam"]["enable_patent_search"] is False
    for c in combos:
        t, b = c["targets"], c["beam"]
        assert 1.45 <= t["density"] <= 1.95
        assert 6500 <= t["det_velocity"] <= 9500
        assert 17.5 <= t["det_pressure"] <= 40.5
        assert -130 <= t["hf_solid"] <= 100
        assert 3 <= b["depth"] <= 8 and 3 <= b["width"] <= 8 and 2 <= b["top_k"] <= 5
        TargetProperties(**t)
        BeamParameters(**{**cfg["fixed_beam"], **b})
    d = np.array([c["targets"]["density"] for c in combos])
    assert np.corrcoef(d, [c["targets"]["det_velocity"] for c in combos])[0, 1] > 0.8
    assert np.corrcoef(d, [c["targets"]["det_pressure"] for c in combos])[0, 1] > 0.8


def test_generator_deterministic_and_matches_committed():
    assert gen.generate(500) == gen.generate(500)
    assert gen.generate(500) == json.loads((ROOT / "configs" / "benchmark_runs_500.json").read_text())
    assert gen.generate(20, rng_seed=1) != gen.generate(20, rng_seed=2)


@pytest.mark.parametrize("args,expected", [([], 100), (["--suite", "50"], 100), (["--suite", "500"], 1000)])
def test_suite_flag_selects_config(tmp_path, capsys, args, expected):
    assert rb.main([*args, "--dry-run", "--log-dir", str(tmp_path)]) == 0
    assert f"pending={expected} " in capsys.readouterr().out


def test_config_flag_overrides_suite(tmp_path, capsys):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(gen.generate(3)))
    assert rb.main(["--suite", "500", "--config", str(p), "--dry-run", "--log-dir", str(tmp_path / "l")]) == 0
    assert "pending=6 " in capsys.readouterr().out
