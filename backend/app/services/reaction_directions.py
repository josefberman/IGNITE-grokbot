"""Documented reaction → property-direction mapping and gap-based ranking.

The vectors come from ``docs/reaction_families.md`` (the Direction column).
Each component is a design label in ``{+1, 0, -1}``, not a measured delta:

- ``+1`` / ``"raise"`` — reaction is labeled to increase the property
- ``-1`` / ``"lower"`` — reaction is labeled to decrease the property
- ``0`` / ``"neutral"`` — docs explicitly label no expected change
- ``None`` / ``"unknown"`` — docs are silent or reagents in a family disagree

Family-level consensus is unanimous-only. Mixed signs inside a family are
``unknown``; ranking at run time uses the reagent (specific reaction), not the
broad family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from backend.app.domain.models import PROPERTY_KEYS
from ml.planner_gap import PROPERTY_SCALES, RELATIVE_THRESHOLDS

# (density, det_velocity, det_pressure, hf_solid). None = unknown / silent.
DirectionVector = Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]

_RAISE, _LOWER, _NEUTRAL = 1, -1, 0

# Explicit copy of docs/reaction_families.md Direction column. Do not fill gaps.
REAGENT_DIRECTION_VECTORS: Dict[str, DirectionVector] = {
    # Nitration
    "nitrate_arom_CH": (1, 1, 1, 0),
    "nitrate_alkyl_CH": (1, 1, 1, 0),
    "nitramine_NH": (1, 1, 1, 1),
    "methyl_to_nitromethyl": (0, 1, 1, 0),
    "halogen_Br_to_nitro": (-1, 1, 1, 1),
    "nitrate_ester_OH": (1, 1, 0, 0),
    "gem_dinitro": (1, 1, 1, 1),
    "trinitromethyl": (1, 1, 0, 1),
    "nitroso_arom": (0, -1, 1, 1),
    # Azidation
    "azide_arom": (0, 0, -1, 1),
    "azide_methyl": (0, 1, -1, 1),
    "amino_to_azide": (-1, 0, 0, 1),
    "halogen_I_to_azide": (-1, 1, 1, 1),
    # Heterocycle
    "tetrazole_CH": (1, 0, 1, 1),
    "triazole_CH": (0, 0, 1, 1),
    "furazan_CH": (1, 1, 1, 1),
    "furoxan_CH": (1, 1, 0, 1),
    "tetrazine_CH": (0, 1, 1, 1),
    "oxadiazole_CH": (1, 0, 0, 1),
    "pyrazole_CH": (0, 0, 0, 1),
    "imidazole_CH": (0, 0, 1, 0),
    "triazine_CH": (0, 1, 0, 1),
    "nitrile_to_tetrazole": (1, 1, 1, 0),
    # Oxidation
    "n_oxide": (0, 1, 1, 1),
    "hydroxyl_CH": (1, -1, 1, -1),
    "methoxy_CH": (1, 0, -1, 0),
    "carboxyl_CH": (0, 0, 1, -1),
    "methylester_CH": (1, 0, -1, -1),
    "amino_to_nitro": (0, 1, 1, -1),
    "hydroperoxy_CH": (1, 0, 0, 1),
    "epoxidize_CC": (1, 0, 1, 0),
    "formyl_CH": (0, -1, 0, -1),
    # Amination
    "cyano_CH": (1, 0, 1, 1),
    "hydrazino_CH": (1, 0, -1, 1),
    "amino_CH": (0, -1, -1, 1),
    "guanidino_CH": (0, 1, 0, 1),
    "amide_CH": (0, -1, 1, -1),
    "isocyanate_CH": (0, 1, -1, 0),
    # Halogenation
    "fluoro_arom_CH": (1, 1, 1, -1),
    "trifluoromethyl_CH": (1, 1, 0, -1),
    "chloro_arom_CH": (1, 1, -1, -1),
    "bromo_arom_CH": (1, 1, 0, 0),
    "iodo_arom_CH": (1, 0, 0, 0),
    "difluoramino_CH": (1, 1, 1, 0),
    "fluoro_alkyl": (1, 0, 1, -1),
    # Dehydrogenation
    "dehydrogenate_CC": (0, 1, -1, 1),
    "alkyne_CC": (0, 1, 0, 1),
    # Arylation
    "add_phenyl_CH": (0, 0, 1, 0),
    "benzyl_CH": (0, -1, 1, 0),
    # Reduction
    "remove_nitro_arom": (-1, -1, -1, 0),
    "remove_bromo_arom": (-1, 0, 0, 0),
    "remove_azide_arom": (0, 0, 0, -1),
    "nitro_to_methyl_arom": (-1, -1, -1, -1),
    "nitramine_to_amine": (0, 0, -1, -1),
    "methyl_to_H": (-1, 0, -1, 1),
    "hydrogenate_CC": (0, -1, 1, -1),
    "nitro_to_amino": (0, -1, -1, 1),
    "azide_to_amino": (-1, 0, 1, -1),
    # Alkylation
    "methyl_CH": (-1, -1, 0, 0),
    "ethyl_CH": (-1, 0, -1, 0),
    "hydroxymethyl_CH": (0, -1, 0, -1),
    # Aromatization
    "aromatize_diene": (0, 1, 0, -1),
}


def sign_label(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    if value > 0:
        return "raise"
    if value < 0:
        return "lower"
    return "neutral"


@dataclass(frozen=True)
class ReagentDirection:
    reagent_id: str
    family: str
    density: Optional[int]
    det_velocity: Optional[int]
    det_pressure: Optional[int]
    hf_solid: Optional[int]

    def component(self, key: str) -> Optional[int]:
        return getattr(self, key)

    def as_tuple(self) -> DirectionVector:
        return (self.density, self.det_velocity, self.det_pressure, self.hf_solid)

    def as_labels(self) -> Dict[str, str]:
        return {k: sign_label(self.component(k)) for k in PROPERTY_KEYS}


def _reagent_catalog() -> Dict[str, ReagentDirection]:
    from backend.app.services.reaction_primitives import list_primitives

    by_id = {p["family_id"]: p for p in list_primitives() if p["family_id"] != "identity"}
    out: Dict[str, ReagentDirection] = {}
    for rid, vec in REAGENT_DIRECTION_VECTORS.items():
        prim = by_id.get(rid)
        family = prim["group"] if prim else "unknown"
        out[rid] = ReagentDirection(rid, family, *vec)
    return out


REAGENT_DIRECTIONS: Dict[str, ReagentDirection] = _reagent_catalog()


def family_direction_consensus() -> Dict[str, Dict[str, Optional[int]]]:
    """Unanimous family×property sign; ``None`` when reagents disagree or any is unknown."""
    grouped: Dict[str, List[ReagentDirection]] = {}
    for row in REAGENT_DIRECTIONS.values():
        grouped.setdefault(row.family, []).append(row)
    consensus: Dict[str, Dict[str, Optional[int]]] = {}
    for family, rows in grouped.items():
        props: Dict[str, Optional[int]] = {}
        for key in PROPERTY_KEYS:
            signs = {row.component(key) for row in rows}
            if None in signs or len(signs) != 1:
                props[key] = None
            else:
                props[key] = next(iter(signs))
        consensus[family] = props
    return consensus


def desired_property_signs(
    parent: Mapping[str, float],
    target: Mapping[str, float],
) -> Dict[str, int]:
    """``+1`` if the parent is below target, ``-1`` if above, ``0`` if within deadband."""
    out: Dict[str, int] = {}
    for key in PROPERTY_KEYS:
        gap = float(target[key]) - float(parent.get(key, target[key]))
        scale = PROPERTY_SCALES[key]
        deadband = max(abs(scale * RELATIVE_THRESHOLDS[key]), 1e-6)
        if abs(gap) <= deadband:
            out[key] = 0
        elif gap > 0:
            out[key] = 1
        else:
            out[key] = -1
    return out


def gap_weights(
    parent: Mapping[str, float],
    target: Mapping[str, float],
) -> Dict[str, float]:
    """Per-property weight in ``[0, 1]`` from the scaled absolute gap (0 in the deadband)."""
    out: Dict[str, float] = {}
    for key in PROPERTY_KEYS:
        gap = float(target[key]) - float(parent.get(key, target[key]))
        scale = PROPERTY_SCALES[key]
        deadband = max(abs(scale * RELATIVE_THRESHOLDS[key]), 1e-6)
        if abs(gap) <= deadband:
            out[key] = 0.0
        else:
            out[key] = min(1.0, abs(gap) / scale)
    return out


def reaction_alignment_score(
    vector: DirectionVector,
    parent: Mapping[str, float],
    target: Mapping[str, float],
) -> float:
    """Dot product of documented signs with the desired gap, skipping unknown axes.

    A reagent that raises density scores positive when density is below target
    and negative when density is above target. Neutral (0) and unknown (None)
    components contribute 0 — unknown is not treated as raise or lower.
    """
    desired = desired_property_signs(parent, target)
    weights = gap_weights(parent, target)
    score = 0.0
    for i, key in enumerate(PROPERTY_KEYS):
        sign = vector[i]
        if sign is None:
            continue
        score += float(sign) * float(desired[key]) * weights[key]
    return score


def rank_reagents_for_gap(
    parent_props: Mapping[str, float],
    targets: Mapping[str, float],
    *,
    parent_smiles: Optional[str] = None,
    skip: Optional[Iterable[str]] = None,
    candidates: Optional[Sequence[str]] = None,
) -> List[str]:
    """Rank specific reactions by how well their documented direction closes the gap.

    Applicable reagents (SMARTS match on ``parent_smiles``) are preferred at equal
    score. Unknown axes are skipped, never guessed.
    """
    skip_set = set(skip or ())
    pool = list(candidates) if candidates is not None else list(REAGENT_DIRECTION_VECTORS)
    pool = [rid for rid in pool if rid not in skip_set and rid in REAGENT_DIRECTION_VECTORS]

    applicable: set[str] = set()
    if parent_smiles:
        from backend.app.services.reaction_primitives import matching_reaction_ids

        applicable = set(matching_reaction_ids(parent_smiles))

    scored: List[Tuple[float, int, str]] = []
    for rid in pool:
        score = reaction_alignment_score(
            REAGENT_DIRECTION_VECTORS[rid], parent_props, targets
        )
        prefer = 0 if (not applicable or rid in applicable) else 1
        scored.append((score, prefer, rid))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [rid for _, _, rid in scored]


def ambiguous_family_cases() -> List[Dict[str, str]]:
    """Every family×property whose reagents do not share one documented sign."""
    rows: List[Dict[str, str]] = []
    consensus = family_direction_consensus()
    for family in sorted(consensus):
        for key in PROPERTY_KEYS:
            if consensus[family][key] is None:
                signs = sorted(
                    {
                        sign_label(REAGENT_DIRECTIONS[rid].component(key))
                        for rid, row in REAGENT_DIRECTIONS.items()
                        if row.family == family
                    }
                )
                rows.append(
                    {
                        "family": family,
                        "property": key,
                        "consensus": "unknown",
                        "reagent_labels": ",".join(signs),
                    }
                )
    return rows
