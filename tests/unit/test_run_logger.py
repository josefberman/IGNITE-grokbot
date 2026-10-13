"""Search-run JSON logger."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from backend.app.domain.models import (
    CandidateMolecule,
    ModificationMode,
    ModificationRecord,
    PredictionSource,
    PropertyEstimate,
    SafetyFinding,
    ScoreBreakdown,
    TargetProperties,
)
from backend.app.services.run_logger import RunLogger, build_run_log


def _candidate(smiles: str, *, generation: int, total: float, mape: float) -> CandidateMolecule:
    return CandidateMolecule(
        smiles=smiles,
        canonical_smiles=smiles,
        generation=generation,
        properties={
            "density": PropertyEstimate(value=1.7, source=PredictionSource.XGBOOST),
            "det_velocity": PropertyEstimate(value=8000.0, source=PredictionSource.XGBOOST),
            "det_pressure": PropertyEstimate(value=28.0, source=PredictionSource.XGBOOST),
            "hf_solid": PropertyEstimate(value=80.0, source=PredictionSource.XGBOOST),
        },
        mape=mape,
        score=ScoreBreakdown(
            mape=mape,
            mape_normalized=mape / 100.0,
            sascore_penalty=0.2,
            minimal_edit_penalty=0.1,
            total=total,
            weights={"mape": 0.65, "sascore": 0.25, "minimal_edit": 0.10},
        ),
        safety=SafetyFinding(score=0.2, sascore=2.4),
        modification=ModificationRecord(
            reaction_family="nitration",
            reagent="nitrate_arom_CH",
            edit_distance=2,
            planner="rule_based",
        )
        if generation
        else None,
        parent_smiles="c1ccccc1" if generation else None,
    )


def test_build_run_log_ranks_candidates_and_keeps_scores():
    seed = _candidate("c1ccccc1", generation=0, total=0.9, mape=20.0)
    child = _candidate("c1ccccc1N(=O)=O", generation=1, total=0.4, mape=8.0)
    targets = TargetProperties(density=1.8, det_velocity=8500, det_pressure=30, hf_solid=100)
    record = build_run_log(
        job_id="job-1",
        mode=ModificationMode.RULE_BASED,
        targets=targets,
        seed_smiles=[seed.canonical_smiles],
        candidates=[seed, child],
        timestamp=datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc),
        status="completed",
    )
    assert record["job_id"] == "job-1"
    assert record["timestamp"] == "2026-10-13T12:00:00+00:00"
    assert record["mode"] == "rule_based"
    assert record["target_properties"]["density"] == 1.8
    assert record["seed_smiles"] == ["c1ccccc1"]
    assert [c["smiles"] for c in record["candidates"]] == [
        "c1ccccc1N(=O)=O",
        "c1ccccc1",
    ]
    best = record["candidates"][0]
    assert best["rank"] == 1
    assert best["predicted_properties"]["density"] == 1.7
    assert best["fit"]["mape"] == 8.0
    assert best["synthesis_ease"]["sascore"] == 2.4
    assert best["synthesis_ease"]["sascore_penalty"] == 0.2
    assert best["change_size"]["edit_distance"] == 2
    assert best["overall"]["score"] == 0.4
    assert best["overall"]["rank"] == 1


def test_run_logger_writes_json_file(tmp_path):
    logger = RunLogger(tmp_path, enabled=True)
    seed = _candidate("C", generation=0, total=0.5, mape=10.0)
    path = logger.log_run(
        job_id="abc",
        mode="planner",
        targets={"density": 1.8, "det_velocity": 8500, "det_pressure": 30, "hf_solid": 100},
        seed_smiles=["C"],
        candidates=[seed],
        status="completed",
    )
    assert path is not None
    assert path.parent == tmp_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_id"] == "abc"
    assert payload["mode"] == "planner"
    assert payload["candidates"][0]["smiles"] == "C"


def test_run_logger_disabled_writes_nothing(tmp_path):
    logger = RunLogger(tmp_path, enabled=False)
    path = logger.log_run(
        job_id="abc",
        mode="rule_based",
        targets={"density": 1.8, "det_velocity": 8500, "det_pressure": 30, "hf_solid": 100},
        seed_smiles=["C"],
        candidates=[],
    )
    assert path is None
    assert list(tmp_path.iterdir()) == []
