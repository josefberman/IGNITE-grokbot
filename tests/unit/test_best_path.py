"""best_path construction in run logs (mode-agnostic + planner/rule_based records)."""

from __future__ import annotations

import pytest

from backend.app.domain.models import (
    CandidateMolecule,
    ModificationMode,
    ModificationRecord,
    PredictionSource,
    PropertyEstimate,
    SafetyFinding,
    ScoreBreakdown,
)
from backend.app.services.run_logger import build_best_path, build_run_log

TARGETS = {"density": 1.8, "det_velocity": 8500, "det_pressure": 30, "hf_solid": 100}


def _cand(smiles, parent=None, *, total, mod=None, link_by_id=True):
    return CandidateMolecule(
        smiles=smiles,
        canonical_smiles=smiles,
        generation=(parent.generation + 1) if parent else 0,
        parent_id=parent.id if (parent and link_by_id) else None,
        parent_smiles=parent.canonical_smiles if parent else None,
        properties={
            k: PropertyEstimate(value=v, source=PredictionSource.XGBOOST)
            for k, v in {"density": 1.7, "det_velocity": 8000.0, "det_pressure": 28.0, "hf_solid": 80.0}.items()
        },
        mape=total * 10,
        score=ScoreBreakdown(
            mape=total * 10, mape_normalized=total / 10, sascore_penalty=0.2,
            minimal_edit_penalty=0.1, total=total, weights={},
        ),
        safety=SafetyFinding(score=0.1, sascore=2.5),
        modification=mod,
    )


def _rule(reagent):
    return ModificationRecord(reaction_family=reagent, reagent=reagent, edit_distance=2, planner="rule_based")


def _learned(family, reagent):
    return ModificationRecord(reaction_family=family, reagent=reagent, edit_distance=3, planner="learned")


@pytest.mark.parametrize(
    "mode,mk",
    [
        (ModificationMode.RULE_BASED, lambda i: _rule(f"reagent_{i}")),
        (ModificationMode.PLANNER, lambda i: _learned(f"family_{i}", f"reagent_{i}")),
    ],
)
def test_best_path_chain_both_modes(mode, mk):
    seed = _cand("C", total=0.9)
    a = _cand("CN", seed, total=0.7, mod=mk(1))
    sibling = _cand("CO", seed, total=0.8, mod=mk(9))
    b = _cand("CNN", a, total=0.5, mod=mk(2))
    c = _cand("CNN=O", b, total=0.3, mod=mk(3))
    record = build_run_log(
        job_id="j", mode=mode, targets=TARGETS, seed_smiles=["C"],
        candidates=[c, sibling, seed, b, a], best=c,
    )
    path = record["best_path"]
    assert [s["smiles"] for s in path] == ["C", "CN", "CNN", "CNN=O"]
    assert [s["step"] for s in path] == [0, 1, 2, 3]
    assert path[0]["edit"] is None
    for i, step in enumerate(path[1:], 1):
        assert step["edit"]["reagent"] == f"reagent_{i}"
        expected_family = f"reagent_{i}" if mode == ModificationMode.RULE_BASED else f"family_{i}"
        assert step["edit"]["reaction_family"] == expected_family
        assert set(step["predicted_properties"]) == {"density", "det_velocity", "det_pressure", "hf_solid"}
        assert step["fit"]["mape"] is not None
        assert step["synthesis_ease"]["sascore"] == 2.5
        assert step["change_size"]["edit_distance"] >= 2
        assert step["overall"]["score"] is not None


def test_best_path_falls_back_to_parent_smiles_and_default_best():
    seed = _cand("C", total=0.9)
    child = _cand("CN", seed, total=0.2, mod=_rule("amine"), link_by_id=False)
    record = build_run_log(job_id="j", mode="rule_based", targets=TARGETS, seed_smiles=["C"], candidates=[seed, child])
    assert [s["smiles"] for s in record["best_path"]] == ["C", "CN"]


def test_best_path_seed_only_and_empty():
    seed = _cand("C", total=0.9)
    path = build_best_path(seed, [seed])
    assert len(path) == 1 and path[0]["edit"] is None and path[0]["step"] == 0
    assert build_best_path(None, []) == []
