"""Durable JSON logs for target-property molecule generation runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from backend.app.domain.models import CandidateMolecule, ModificationMode, TargetProperties

logger = logging.getLogger(__name__)


def _score_payload(candidate: CandidateMolecule) -> Dict[str, Any]:
    score = candidate.score
    mod = candidate.modification
    safety = candidate.safety
    return {
        "smiles": candidate.canonical_smiles,
        "generation": candidate.generation,
        "predicted_properties": candidate.property_values(),
        "fit": {
            "mape": score.mape if score is not None else candidate.mape,
            "mape_normalized": score.mape_normalized if score is not None else None,
        },
        "synthesis_ease": {
            "sascore_penalty": score.sascore_penalty if score is not None else None,
            "sascore": safety.sascore if safety is not None else None,
        },
        "change_size": {
            "minimal_edit_penalty": score.minimal_edit_penalty if score is not None else None,
            "edit_distance": mod.edit_distance if mod is not None else 0,
        },
        "overall": {
            "score": score.total if score is not None else None,
        },
        "reaction_family": mod.reaction_family if mod is not None else None,
        "reagent": mod.reagent if mod is not None else None,
        "planner": mod.planner if mod is not None else ("seed" if candidate.generation == 0 else None),
        "parent_smiles": candidate.parent_smiles,
    }


def build_run_log(
    *,
    job_id: str,
    mode: ModificationMode | str,
    targets: TargetProperties | Mapping[str, float],
    seed_smiles: Sequence[str],
    candidates: Sequence[CandidateMolecule],
    timestamp: Optional[datetime] = None,
    status: Optional[str] = None,
) -> Dict[str, Any]:
    ts = timestamp or datetime.now(timezone.utc)
    target_dict = targets.as_dict() if isinstance(targets, TargetProperties) else dict(targets)
    ranked = sorted(
        candidates,
        key=lambda c: (
            c.score.total if c.score is not None else 1e9,
            c.mape,
        ),
    )
    entries: List[Dict[str, Any]] = []
    for i, cand in enumerate(ranked, 1):
        payload = _score_payload(cand)
        payload["rank"] = i
        payload["overall"]["rank"] = i
        entries.append(payload)
    return {
        "job_id": job_id,
        "timestamp": ts.astimezone(timezone.utc).isoformat(),
        "mode": mode.value if isinstance(mode, ModificationMode) else str(mode),
        "status": status,
        "target_properties": target_dict,
        "seed_smiles": list(seed_smiles),
        "candidates": entries,
    }


class RunLogger:
    """Write one JSON file per search run under ``log_dir``."""

    def __init__(self, log_dir: Path, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.enabled = enabled

    def write(self, record: Mapping[str, Any]) -> Optional[Path]:
        if not self.enabled:
            return None
        self.log_dir.mkdir(parents=True, exist_ok=True)
        job_id = str(record.get("job_id") or "unknown")
        stamp = str(record.get("timestamp") or datetime.now(timezone.utc).isoformat())
        safe_stamp = (
            stamp.replace(":", "").replace("-", "").replace("+", "p").replace(".", "")
        )
        path = self.log_dir / f"{safe_stamp}_{job_id}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        logger.info("Wrote search run log %s", path)
        return path

    def log_run(
        self,
        *,
        job_id: str,
        mode: ModificationMode | str,
        targets: TargetProperties | Mapping[str, float],
        seed_smiles: Sequence[str],
        candidates: Sequence[CandidateMolecule],
        status: Optional[str] = None,
    ) -> Optional[Path]:
        record = build_run_log(
            job_id=job_id,
            mode=mode,
            targets=targets,
            seed_smiles=seed_smiles,
            candidates=candidates,
            status=status,
        )
        return self.write(record)
