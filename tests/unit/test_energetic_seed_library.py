"""Tests for the curated energetic seed library and seed-CSV selection."""

from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from backend.app.core.config import Settings
from backend.app.domain.models import PROPERTY_KEYS, PredictionSource, TargetProperties
from backend.app.services.property_estimator import LiteratureRetriever
from backend.app.services.seed_selector import SeedLibrary

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "seeds" / "energetic_seed_library.csv"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


def test_csv_loads_with_schema(df):
    assert len(df) == 77
    for key in PROPERTY_KEYS:
        for col in (key, f"{key}_source", f"{key}_provenance"):
            assert col in df.columns
    assert df["SMILES"].is_unique


def test_smiles_valid_canonical_neutral(df):
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        assert Chem.MolToSmiles(mol) == smi
        assert "." not in smi
        assert Chem.GetFormalCharge(mol) == 0


def test_provenance_complete(df):
    for key in PROPERTY_KEYS:
        assert df[key].notna().all()
        assert set(df[f"{key}_provenance"]) <= {"literature", "predicted"}
        assert df[f"{key}_source"].fillna("").str.len().gt(10).all()
        lit = df[df[f"{key}_provenance"] == "literature"]
        assert lit[f"{key}_source"].str.contains("http").all()


def test_no_duplicate_structures(df):
    keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(s)) for s in df["SMILES"]]
    assert len(set(keys)) == len(keys)
    assert df["name"].is_unique


def test_no_ionic_fragments(df):
    for smi in df["SMILES"]:
        mol = Chem.MolFromSmiles(smi)
        assert not any(a.GetFormalCharge() and a.GetDegree() == 0 for a in mol.GetAtoms()), smi


def test_expansion_molecules_present(df):
    for name in ["DINGU", "Keto-RDX", "BCHMX", "ANPZ", "LLM-116", "ANTA", "DNTF", "PYX",
                 "1,3-Dinitrobenzene", "Methyl nitrate", "EDNA", "TATP", "HMTD"]:
        assert name in set(df["name"])


def test_provenance_counts(df):
    lit = sum((df[f"{k}_provenance"] == "literature").sum() for k in PROPERTY_KEYS)
    pred = sum((df[f"{k}_provenance"] == "predicted").sum() for k in PROPERTY_KEYS)
    assert lit + pred == 4 * len(df)
    assert lit == 105


def test_expected_molecules_present(df):
    for name in ["TNT", "RDX", "HMX", "CL-20", "PETN", "TATB", "FOX-7", "NTO", "LLM-105",
                 "TNAZ", "Nitroglycerin", "Picric acid", "DNAN"]:
        assert name in set(df["name"])


def test_default_seed_csv_is_energetic_library(monkeypatch):
    monkeypatch.delenv("IGNITE_SEED_CSV", raising=False)
    assert Settings().seed_csv.resolve() == CSV.resolve()


def test_seed_csv_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom.csv"
    pd.read_csv(CSV).head(3).to_csv(custom, index=False)
    monkeypatch.setenv("IGNITE_SEED_CSV", str(custom))
    s = Settings()
    assert s.seed_csv == custom
    assert len(SeedLibrary(s.seed_csv).df) == 3


def test_training_dataset_refused_as_seeds():
    with pytest.raises(ValueError):
        SeedLibrary(ROOT / "sample_start_molecules.csv")


def test_predicted_values_not_treated_as_literature():
    lib = SeedLibrary(CSV)
    lit = LiteratureRetriever(enabled=True)
    lit.load_seed_library(lib.df.to_dict(orient="records"))
    tnt = lib.df[lib.df["name"] == "TNT"].iloc[0]["canonical_smiles"]
    dnan = lib.df[lib.df["name"] == "DNAN"].iloc[0]["canonical_smiles"]
    assert lit.retrieve(tnt) is not None
    assert lit.retrieve(dnan) is None
    targets = TargetProperties(density=1.58, det_velocity=6500, det_pressure=17, hf_solid=-45)
    picked = {c.canonical_smiles: c for c in lib.select_seeds(targets, 100.0, max_seeds=30)}
    assert picked[dnan].properties["density"].source == PredictionSource.XGBOOST
    assert picked[tnt].properties["density"].source == PredictionSource.DATASET
