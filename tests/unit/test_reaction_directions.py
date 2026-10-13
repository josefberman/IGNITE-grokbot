"""Rule-based direction mapping and gap-based reagent ranking."""

from __future__ import annotations

import re
from pathlib import Path

from backend.app.domain.models import PROPERTY_KEYS
from backend.app.services.reaction_directions import (
    REAGENT_DIRECTION_VECTORS,
    REAGENT_DIRECTIONS,
    ambiguous_family_cases,
    desired_property_signs,
    family_direction_consensus,
    rank_reagents_for_gap,
    reaction_alignment_score,
    sign_label,
)
from backend.app.services.reaction_primitives import list_primitives

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "reaction_families.md"

_ON_TARGET = {
    "density": 1.8,
    "det_velocity": 8500.0,
    "det_pressure": 30.0,
    "hf_solid": 100.0,
}


def _parse_doc_directions() -> dict[str, tuple[int, int, int, int]]:
    text = DOCS.read_text(encoding="utf-8")
    row_re = re.compile(
        r"^\| `([a-zA-Z0-9_]+)` \| .+ \| ([+\-0-9,\s]+) \|\s*$",
        re.MULTILINE,
    )
    out: dict[str, tuple[int, int, int, int]] = {}
    for reagent, raw in row_re.findall(text):
        parts = [int(p.strip()) for p in raw.split(",")]
        assert len(parts) == 4, (reagent, raw)
        out[reagent] = tuple(parts)  # type: ignore[assignment]
    return out


def test_mapping_covers_every_non_identity_primitive():
    prims = {p["family_id"] for p in list_primitives() if p["family_id"] != "identity"}
    assert set(REAGENT_DIRECTION_VECTORS) == prims
    assert set(REAGENT_DIRECTIONS) == prims


def test_mapping_matches_reaction_families_docs():
    documented = _parse_doc_directions()
    assert documented, "docs/reaction_families.md Direction rows were not parsed"
    assert set(documented) == set(REAGENT_DIRECTION_VECTORS)
    for rid, vec in documented.items():
        assert REAGENT_DIRECTION_VECTORS[rid] == vec


def test_mapping_matches_primitive_delta_tuples():
    for prim in list_primitives():
        if prim["family_id"] == "identity":
            continue
        assert REAGENT_DIRECTION_VECTORS[prim["family_id"]] == prim["delta"]


def test_documented_zeros_are_neutral_not_unknown():
    # The docs label 0 explicitly; that is neutral, not a silent guess.
    assert sign_label(0) == "neutral"
    assert sign_label(None) == "unknown"
    assert any(0 in vec for vec in REAGENT_DIRECTION_VECTORS.values())
    assert all(None not in vec for vec in REAGENT_DIRECTION_VECTORS.values())


def test_family_consensus_unknown_when_reagents_disagree():
    consensus = family_direction_consensus()
    nitration = consensus["nitration"]
    # halogen_Br_to_nitro lowers density; most nitrations raise it.
    assert nitration["density"] is None
    aromatization = consensus["aromatization"]
    assert aromatization["det_velocity"] == 1
    assert aromatization["hf_solid"] == -1
    cases = ambiguous_family_cases()
    assert any(c["family"] == "nitration" and c["property"] == "density" for c in cases)
    assert not any(c["family"] == "aromatization" and c["property"] == "det_velocity" for c in cases)


def test_desired_signs_density_below_target():
    parent = {**_ON_TARGET, "density": 1.2}
    signs = desired_property_signs(parent, _ON_TARGET)
    assert signs["density"] == 1
    assert signs["det_velocity"] == 0
    assert signs["det_pressure"] == 0
    assert signs["hf_solid"] == 0


def test_rank_prefers_density_raising_when_density_is_low():
    parent = {**_ON_TARGET, "density": 1.2}
    ranked = rank_reagents_for_gap(parent, _ON_TARGET)
    raise_ids = [r for r in ranked if REAGENT_DIRECTION_VECTORS[r][0] == 1]
    lower_ids = [r for r in ranked if REAGENT_DIRECTION_VECTORS[r][0] == -1]
    assert raise_ids and lower_ids
    last_raise = max(ranked.index(r) for r in raise_ids)
    first_lower = min(ranked.index(r) for r in lower_ids)
    assert last_raise < first_lower
    assert REAGENT_DIRECTION_VECTORS[ranked[0]][0] == 1


def test_rank_prefers_density_lowering_when_density_is_high():
    parent = {**_ON_TARGET, "density": 2.3}
    ranked = rank_reagents_for_gap(parent, _ON_TARGET)
    assert REAGENT_DIRECTION_VECTORS[ranked[0]][0] == -1
    raise_ids = [r for r in ranked if REAGENT_DIRECTION_VECTORS[r][0] == 1]
    lower_ids = [r for r in ranked if REAGENT_DIRECTION_VECTORS[r][0] == -1]
    last_lower = max(ranked.index(r) for r in lower_ids)
    first_raise = min(ranked.index(r) for r in raise_ids)
    assert last_lower < first_raise


def test_unknown_axis_is_not_guessed_as_raise_or_lower():
    parent = {**_ON_TARGET, "density": 1.2}
    # Unknown density must not score as if it raised density.
    unknown = reaction_alignment_score((None, 0, 0, 0), parent, _ON_TARGET)
    raise_density = reaction_alignment_score((1, 0, 0, 0), parent, _ON_TARGET)
    lower_density = reaction_alignment_score((-1, 0, 0, 0), parent, _ON_TARGET)
    assert unknown == 0.0
    assert raise_density > 0
    assert lower_density < 0
    assert unknown != raise_density
    assert unknown != lower_density


def test_skip_and_candidate_filters():
    parent = {**_ON_TARGET, "density": 1.2}
    ranked = rank_reagents_for_gap(
        parent,
        _ON_TARGET,
        skip={"nitrate_arom_CH"},
        candidates=["nitrate_arom_CH", "remove_nitro_arom", "iodo_arom_CH"],
    )
    assert "nitrate_arom_CH" not in ranked
    assert ranked[0] == "iodo_arom_CH"
    assert ranked[-1] == "remove_nitro_arom"


def test_property_key_order_matches_docs_column():
    assert PROPERTY_KEYS == ("density", "det_velocity", "det_pressure", "hf_solid")
