"""Children are compared with the seed without the minimal-edit penalty."""

from backend.app.domain.models import CandidateMolecule, ScoreBreakdown
from backend.app.services.beam_search import BeamSearchController
from backend.app.services.scoring import DEFAULT_WEIGHTS, rank_candidates, seed_comparable_score


def _cand(name, gen, mape, sas, edit):
    w = DEFAULT_WEIGHTS
    total = w["mape"] * mape / 100 + w["sascore"] * sas + w["minimal_edit"] * edit
    return CandidateMolecule(
        smiles=name, canonical_smiles=name, generation=gen, mape=mape,
        score=ScoreBreakdown(mape=mape, mape_normalized=mape / 100, sascore_penalty=sas,
                             minimal_edit_penalty=edit, total=total, weights=dict(w)),
    )


def test_child_with_better_mape_beats_seed_despite_edits():
    # DNAN-like case from the 8-target run: child total is worse only because of edits.
    seed = _cand("seed", 0, 3.89, 0.03, 0.0)
    child = _cand("child", 1, 1.65, 0.035, 0.3)
    assert child.score.total > seed.score.total
    assert seed_comparable_score(child) < seed_comparable_score(seed)
    assert rank_candidates([seed, child])[0] is child


def test_children_still_ranked_with_edit_penalty():
    small = _cand("small-edit", 1, 2.0, 0.03, 0.1)
    big = _cand("big-edit", 1, 1.9, 0.03, 0.5)  # slightly better MAPE, far larger edit
    assert rank_candidates([big, small])[:2] == [small, big]


def test_seed_wins_when_genuinely_better():
    seed = _cand("seed", 0, 2.0, 0.03, 0.0)
    child = _cand("child", 1, 2.5, 0.04, 0.1)
    assert rank_candidates([child, seed])[0] is seed
    tie = _cand("tie", 1, 2.0, 0.03, 0.2)
    assert rank_candidates([tie, seed])[0] is seed


def test_mixed_ordering_and_survivors():
    seed = _cand("seed", 0, 3.0, 0.03, 0.0)
    a = _cand("a", 1, 1.0, 0.03, 0.4)   # beats seed edit-free
    b = _cand("b", 1, 5.0, 0.03, 0.1)   # worse than seed
    order = [c.canonical_smiles for c in rank_candidates([b, seed, a])]
    assert order == ["a", "seed", "b"]
    survivors = BeamSearchController._select_survivors([b, seed, a], 2, explore=False)
    assert [c.canonical_smiles for c in survivors] == ["a", "seed"]
