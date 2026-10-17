"""Seed molecule loading and MAPE-threshold multi-seed selection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from backend.app.domain.models import (
    PROPERTY_KEYS,
    CandidateMolecule,
    PredictionSource,
    PropertyEstimate,
    TargetProperties,
)
from backend.app.services.chemistry import canonicalize, inchikey_of
from backend.app.services.scoring import calculate_mape

logger = logging.getLogger(__name__)

COLUMN_MAP = {
    "density": "density",
    "det_velocity": "det_velocity",
    "det_pressure": "det_pressure",
    "hf_solid": "hf_solid",
}


# XGBoost training data; never a seed library.
TRAINING_DATA_FILENAME = "sample_start_molecules.csv"


class SeedLibrary:
    def __init__(self, csv_path: Path | str):
        self.csv_path = Path(csv_path)
        if self.csv_path.name == TRAINING_DATA_FILENAME:
            raise ValueError(
                f"{TRAINING_DATA_FILENAME} is the XGBoost training dataset and cannot be used as a "
                "seed library; use data/seeds/energetic_seed_library.csv (IGNITE_SEED_CSV)."
            )
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "SMILES" not in df.columns:
            raise ValueError("Seed CSV must contain SMILES column")
        # Drop unnamed index columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        records = []
        seen = set()
        for idx, row in df.iterrows():
            smiles = str(row["SMILES"])
            can = canonicalize(smiles)
            if not can or can in seen:
                continue
            props = {}
            ok = True
            for key in PROPERTY_KEYS:
                val = row.get(key)
                if pd.isna(val):
                    ok = False
                    break
                props[key] = float(val)
            if not ok:
                continue
            seen.add(can)
            prov = {}
            for key in PROPERTY_KEYS:
                col = f"{key}_provenance"
                if col in df.columns and not pd.isna(row.get(col)):
                    prov[col] = str(row[col]).strip().lower()
            if "name" in df.columns and not pd.isna(row.get("name")):
                prov["name"] = str(row["name"])
            records.append(
                {
                    **prov,
                    "source_row": int(idx),
                    "smiles": smiles,
                    "canonical_smiles": can,
                    "inchikey": inchikey_of(can),
                    **props,
                }
            )
        out = pd.DataFrame(records)
        logger.info("Loaded %d unique complete seed molecules", len(out))
        return out

    def select_seeds(
        self,
        targets: TargetProperties,
        mape_threshold: float,
        max_seeds: int = 5,
    ) -> List[CandidateMolecule]:
        target = targets.as_dict()
        scored: List[Tuple[float, CandidateMolecule]] = []
        for _, row in self.df.iterrows():
            props = {k: float(row[k]) for k in PROPERTY_KEYS}
            mape = calculate_mape(props, target)
            if mape > mape_threshold and scored:
                # still collect absolute best if none under threshold
                pass
            cand = CandidateMolecule(
                smiles=row["smiles"],
                canonical_smiles=row["canonical_smiles"],
                inchikey=row.get("inchikey"),
                generation=0,
                properties={
                    k: PropertyEstimate(
                        value=props[k],
                        source=(
                            PredictionSource.XGBOOST
                            if row.get(f"{k}_provenance") == "predicted"
                            else PredictionSource.DATASET
                        ),
                        confidence=0.65 if row.get(f"{k}_provenance") == "predicted" else 0.9,
                    )
                    for k in PROPERTY_KEYS
                },
                mape=mape,
                lineage=[],
            )
            scored.append((mape, cand))

        scored.sort(key=lambda x: x[0])
        under = [c for m, c in scored if m <= mape_threshold]
        if under:
            return under[:max_seeds]
        # Fallback: closest seeds even if above threshold
        return [c for _, c in scored[:max_seeds]]

    def all_canonical_smiles(self) -> List[str]:
        return list(self.df["canonical_smiles"].astype(str))
