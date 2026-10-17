"""Integration tests for seed selection and bounded beam search."""

from pathlib import Path

import pytest

from backend.app.domain.models import BeamParameters, ModificationMode, TargetProperties
from backend.app.services.beam_search import BeamSearchController
from backend.app.services.modification import ModificationMechanism
from backend.app.services.novelty import NoveltyService
from backend.app.services.prediction import PropertyPredictor
from backend.app.services.property_estimator import LiteratureRetriever, PropertyEstimator
from backend.app.services.seed_selector import SeedLibrary
from backend.app.services.structure import smiles_to_png

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def seeds():
    return SeedLibrary(ROOT / "data" / "seeds" / "energetic_seed_library.csv")


@pytest.fixture(scope="module")
def controller(seeds):
    predictor = PropertyPredictor(ROOT / "models")
    literature = LiteratureRetriever(enabled=True)
    literature.load_seed_library(seeds.df.to_dict(orient="records"))
    estimator = PropertyEstimator(predictor, literature)
    novelty = NoveltyService(
        seed_smiles=seeds.all_canonical_smiles()[:500],
        pubchem_enabled=False,
        openalex_enabled=False,
    )
    return BeamSearchController(seeds, estimator, novelty, ModificationMechanism())


def test_seed_selection_returns_candidates(seeds):
    targets = TargetProperties(density=1.8, det_velocity=8500, det_pressure=32, hf_solid=50)
    selected = seeds.select_seeds(targets, mape_threshold=50.0, max_seeds=3)
    assert 1 <= len(selected) <= 3
    assert selected[0].mape <= selected[-1].mape


def test_beam_search_bounded(controller):
    targets = TargetProperties(density=1.7, det_velocity=7500, det_pressure=25, hf_solid=20)
    beam = BeamParameters(
        depth=2,
        width=3,
        top_k=3,
        mape_threshold=0.5,
        random_seed=0,
        enable_literature=True,
        enable_patent_search=False,
        early_stop_on_mape=False,
    )
    best, top, progress, beam_tree = controller.run(
        "test-job", targets, beam, mode=ModificationMode.RULE_BASED
    )
    assert controller.last_run_log is not None
    assert controller.last_run_log["mode"] == "rule_based"
    assert controller.last_run_log["seed_smiles"]
    assert controller.last_run_log["candidates"]
    assert best.canonical_smiles
    assert len(beam_tree.nodes) >= 1
    assert len(beam_tree.beam_layers) >= 1
    assert beam_tree.best_id == best.id
    assert best.score is not None
    assert len(top) >= 1
    assert progress.status.value in ("completed", "cancelled")
    assert progress.candidates_generated > 0
    assert any(c.generation > 0 for c in top) or best.generation > 0


def test_estimator_falls_back_to_xgboost(seeds):
    predictor = PropertyPredictor(ROOT / "models")
    lit = LiteratureRetriever(enabled=True)
    lit.load_seed_library(seeds.df.head(20).to_dict(orient="records"))
    est = PropertyEstimator(predictor, lit)
    props = est.estimate("CCCCCCCC", prefer_literature=True)
    assert props is not None
    assert all(p.source.value == "xgboost" for p in props.values())


def test_estimator_exact_seed_is_literature(seeds):
    predictor = PropertyPredictor(ROOT / "models")
    lit = LiteratureRetriever(enabled=True)
    lit.load_seed_library(seeds.df.head(50).to_dict(orient="records"))
    smi = str(seeds.df.iloc[0]["canonical_smiles"])
    est = PropertyEstimator(predictor, lit)
    props = est.estimate(smi, prefer_literature=True)
    assert props is not None
    assert all(p.source.value == "literature" for p in props.values())


def test_structure_png():
    png = smiles_to_png("c1ccccc1", size=128)
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def _assert_best_path(controller, best):
    path = controller.last_run_log["best_path"]
    assert path, "best_path missing"
    assert path[0]["step"] == 0 and path[0]["edit"] is None
    assert path[0]["smiles"] == controller.last_run_log["seed_smiles"][0]
    assert path[-1]["smiles"] == best.canonical_smiles
    assert len(path) == best.generation + 1
    for prev, step in zip(path, path[1:]):
        assert step["parent_smiles"] == prev["smiles"]
        assert step["edit"] is not None and step["edit"]["reagent"]
    return path


def test_beam_search_rule_based_best_path(controller):
    targets = TargetProperties(density=1.8, det_velocity=8500, det_pressure=30, hf_solid=100)
    beam = BeamParameters(depth=3, width=3, top_k=3, mape_threshold=0.5, random_seed=1,
                          enable_literature=False, enable_patent_search=False, early_stop_on_mape=False)
    best, *_ = controller.run("bp-rule", targets, beam, mode=ModificationMode.RULE_BASED)
    path = _assert_best_path(controller, best)
    for step in path[1:]:
        assert step["edit"]["planner"] == "rule_based"
        assert step["edit"]["reaction_family"] == step["edit"]["reagent"]


def test_beam_search_planner_best_path_with_stub(seeds):
    """Planner mode with the learned planner stubbed (torch is optional)."""
    from backend.app.domain.models import ModificationRecord

    class _StubLearned:
        available = True
        unavailable_reason = None

    class _StubPlanner(ModificationMechanism):
        def __init__(self):
            super().__init__()
            self.learned = _StubLearned()

        def expand(self, parent_smiles, parent_props, targets, n, *, mode=None, **kw):
            out = super().expand(parent_smiles, parent_props, targets, n,
                                 mode=ModificationMode.RULE_BASED, **kw)
            return [
                (smi, ModificationRecord(reaction_family=f"group:{m.reagent}", reagent=m.reagent,
                                         smarts=m.smarts, edit_distance=m.edit_distance, planner="learned"))
                for smi, m in out
            ]

    predictor = PropertyPredictor(ROOT / "models")
    estimator = PropertyEstimator(predictor, LiteratureRetriever(enabled=False))
    novelty = NoveltyService(seed_smiles=[], pubchem_enabled=False, openalex_enabled=False)
    ctrl = BeamSearchController(seeds, estimator, novelty, _StubPlanner())
    targets = TargetProperties(density=1.8, det_velocity=8500, det_pressure=30, hf_solid=100)
    beam = BeamParameters(depth=2, width=3, top_k=3, mape_threshold=0.5, random_seed=0,
                          enable_literature=False, enable_patent_search=False, early_stop_on_mape=False)
    best, *_ = ctrl.run("bp-planner", targets, beam, mode=ModificationMode.PLANNER)
    assert ctrl.last_run_log["mode"] == "planner"
    path = _assert_best_path(ctrl, best)
    for step in path[1:]:
        assert step["edit"]["planner"] == "learned"
        assert step["edit"]["reaction_family"].startswith("group:")
