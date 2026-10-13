"""Rule-based planner fallback when torch and/or the checkpoint are missing."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from backend.app.domain.models import ModificationMode
from backend.app.services.modification import (
    LearnedPlannerBundle,
    ModificationMechanism,
    PlannerUnavailableError,
)
from backend.app.services.reaction_primitives import list_primitives

_PARENT = "c1ccccc1N(=O)=O"
_PROPS = {
    "density": 1.5,
    "det_velocity": 7000.0,
    "det_pressure": 20.0,
    "hf_solid": 50.0,
}
_TARGETS = {
    "density": 1.8,
    "det_velocity": 8500.0,
    "det_pressure": 30.0,
    "hf_solid": 100.0,
}


def test_learned_bundle_unavailable_without_checkpoint(tmp_path: Path):
    bundle = LearnedPlannerBundle(tmp_path)
    assert bundle.available is False
    assert bundle._model is None
    assert "checkpoint not found" in bundle.unavailable_reason


def test_learned_bundle_unavailable_when_torch_missing(tmp_path: Path, monkeypatch):
    (tmp_path / "planner.pt").write_bytes(b"placeholder-checkpoint")
    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("simulated missing torch")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    bundle = LearnedPlannerBundle(tmp_path)
    assert bundle.available is False
    assert bundle._model is None
    assert "torch is not installed" in bundle.unavailable_reason


def test_expand_rule_based_without_checkpoint():
    mech = ModificationMechanism(checkpoint_dir=None)
    assert mech.learned.available is False
    out = mech.expand(_PARENT, _PROPS, _TARGETS, n=3, mode=ModificationMode.RULE_BASED)
    assert len(out) >= 1
    known = {p["family_id"] for p in list_primitives() if p["family_id"] != "identity"}
    for _, rec in out:
        assert rec.planner == "rule_based"
        assert rec.reagent
        assert rec.reagent in known or rec.reaction_family


def test_expand_planner_does_not_fall_back_without_checkpoint():
    mech = ModificationMechanism(checkpoint_dir=None)
    with pytest.raises(PlannerUnavailableError, match="checkpoint"):
        mech.expand(_PARENT, _PROPS, _TARGETS, n=3, mode=ModificationMode.PLANNER)
