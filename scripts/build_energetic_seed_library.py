"""Build data/seeds/energetic_seed_library.csv: well-known neutral energetic molecules.

Every literature value below was checked against the cited source (Oct 2026).
Anything without a checked source is filled with the repo XGBoost models and
marked provenance='predicted'. Units match sample_start_molecules.csv /
PROPERTY_UNITS: density g/cm^3, det_velocity m/s, det_pressure GPa, hf_solid kJ/mol.

Run:  python scripts/build_energetic_seed_library.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.domain.models import PROPERTY_KEYS  # noqa: E402
from backend.app.services.prediction import PropertyPredictor  # noqa: E402

OUT = ROOT / "data" / "seeds" / "energetic_seed_library.csv"

KRAUSE = (
    "H. H. Krause, 'New Energetic Materials', in U. Teipel (ed.), Energetic Materials, "
    "Wiley-VCH 2005, ch. 1, Table 1.2 (values at TMD; D and P_CJ are code-calculated; "
    "dHf given in kcal/kg, converted to kJ/mol) "
    "https://application.wiley-vch.de/books/sample/3527302409_c01.pdf"
)
KOCH_LLM105 = (
    "E.-C. Koch, MSIAC report L-159 (2009), ANPZO/LLM-105 (D, P_CJ calculated at TMD) "
    "https://www.msiac.nato.int/publication/l-159-insensitive-explosive-materials-2-26-diamino-35-dinitro-14-pyrazine-1-oxide-anpzo/"
)


def wiki(page: str, note: str = "") -> str:
    s = f"Wikipedia '{page}' chembox (and its cited references) https://en.wikipedia.org/wiki/{page}"
    return f"{s}; {note}" if note else s


def krause(kcal_per_kg: float):
    """Marker: hf in kcal/kg from Krause Table 1.2, converted once MW is known."""
    return ("kcal/kg", kcal_per_kg)


# name, SMILES, {prop: (value, citation)}
MOLECULES = [
    ("TNT", "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.654, KRAUSE), "det_velocity": (6881, KRAUSE), "det_pressure": (19.53, KRAUSE), "hf_solid": (krause(-70.5), KRAUSE)}),
    ("RDX", "C1N(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.816, KRAUSE), "det_velocity": (8977, KRAUSE), "det_pressure": (35.17, KRAUSE), "hf_solid": (krause(72.0), KRAUSE)}),
    ("HMX", "C1N(CN(CN(CN1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.910, KRAUSE), "det_velocity": (9320, KRAUSE), "det_pressure": (39.63, KRAUSE), "hf_solid": (krause(60.5), KRAUSE)}),
    ("PETN", "C(C(CO[N+](=O)[O-])(CO[N+](=O)[O-])CO[N+](=O)[O-])O[N+](=O)[O-]",
     {"density": (1.778, KRAUSE), "det_velocity": (8564, KRAUSE), "det_pressure": (31.39, KRAUSE), "hf_solid": (krause(-407.4), KRAUSE)}),
    ("TATB", "c1(c(c(c(c(c1[N+](=O)[O-])N)[N+](=O)[O-])N)[N+](=O)[O-])N",
     {"density": (1.937, KRAUSE), "det_velocity": (8114, KRAUSE), "det_pressure": (31.15, KRAUSE), "hf_solid": (krause(-129.38), KRAUSE)}),
    ("HNS", "O=[N+]([O-])c1cc([N+](=O)[O-])c(/C=C/c2c([N+](=O)[O-])cc([N+](=O)[O-])cc2[N+](=O)[O-])c([N+](=O)[O-])c1",
     {"density": (1.745, KRAUSE), "det_velocity": (7241, KRAUSE), "det_pressure": (23.40, KRAUSE), "hf_solid": (krause(41.53), KRAUSE)}),
    ("NTO", "C1(=NC(=O)NN1)[N+](=O)[O-]",
     {"density": (1.930, KRAUSE), "det_velocity": (8558, KRAUSE), "det_pressure": (31.12, KRAUSE), "hf_solid": (krause(-237.8), KRAUSE)}),
    ("TNAZ", "C1C(CN1[N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.840, KRAUSE), "det_velocity": (9006, KRAUSE), "det_pressure": (36.37, KRAUSE), "hf_solid": (krause(45.29), KRAUSE)}),
    ("CL-20", "[O-][N+](=O)N1C2C3N(C4C(N3[N+]([O-])=O)N(C(C1N4[N+]([O-])=O)N2[N+]([O-])=O)[N+]([O-])=O)[N+]([O-])=O",
     {"density": (2.044, KRAUSE), "det_velocity": (10065, KRAUSE), "det_pressure": (48.23, KRAUSE), "hf_solid": (krause(220.0), KRAUSE)}),
    ("FOX-7", "NC(N)=C([N+]([O-])=O)[N+]([O-])=O",
     {"density": (1.885, KRAUSE), "det_velocity": (9044, KRAUSE), "det_pressure": (36.05, KRAUSE), "hf_solid": (krause(-85.77), KRAUSE)}),
    ("LLM-105", "Nc1nc(N)[n+]([O-])c([N+](=O)[O-])c1[N+](=O)[O-]",
     {"density": (1.913, KOCH_LLM105), "det_velocity": (8560, KOCH_LLM105), "det_pressure": (33.4, KOCH_LLM105)}),
    ("Nitroglycerin", "C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
     {"density": (1.5931, wiki("Nitroglycerin")), "det_velocity": (7820, wiki("Nitroglycerin")),
      "hf_solid": (-370.0, wiki("Nitroglycerin", "dHf of the condensed (liquid) phase; NG is liquid at 298 K"))}),
    ("Picric acid", "O=[N+]([O-])c1cc(cc([N+]([O-])=O)c1O)[N+]([O-])=O",
     {"density": (1.763, wiki("Picric_acid")), "det_velocity": (7480, wiki("Picric_acid", "D at rho=1.70 g/cm3")),
      "hf_solid": (-51.3 * 4.184, wiki("Picric_acid", "dHf -51.3 kcal/mol converted to kJ/mol"))}),
    ("1,3,5-Trinitrobenzene", "C1=C(C=C(C=C1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.76, wiki("1,3,5-Trinitrobenzene"))}),
    ("Tetryl", "CN(c1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.73, wiki("Tetryl")), "det_velocity": (7570, wiki("Tetryl"))}),
    ("Nitroguanidine", "NC(N)=N[N+]([O-])=O",
     {"density": (1.77, wiki("Nitroguanidine")), "det_velocity": (7650, wiki("Nitroguanidine", "D at sp. gr. 1.5"))}),
    ("EGDN", "C(CO[N+](=O)[O-])O[N+](=O)[O-]",
     {"density": (1.4918, wiki("Ethylene_glycol_dinitrate")), "det_velocity": (7300, wiki("Ethylene_glycol_dinitrate", "Fettaka & Lefebvre, Cent. Eur. J. Energ. Mater."))}),
    ("Benzotrifuroxan", "C12=NO[N+](=C1C3=NO[N+](=C3C4=NO[N+](=C24)[O-])[O-])[O-]",
     {"hf_solid": (606.0, wiki("Benzotrifuroxan", "molar enthalpy of formation (Rouse et al.)"))}),
    ("Erythritol tetranitrate", "C(C(C(CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
     {"density": (1.7219, wiki("Erythritol_tetranitrate")), "det_velocity": (8200, wiki("Erythritol_tetranitrate"))}),
    ("Mannitol hexanitrate", "C([C@H]([C@H]([C@@H]([C@@H](CO[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-])O[N+](=O)[O-]",
     {"density": (1.73, wiki("Mannitol_hexanitrate")), "det_velocity": (8260, wiki("Mannitol_hexanitrate", "D at 1.73 g/cm3"))}),
    ("Hexanitrobenzene", "c1(c(c(c(c(c1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.985, wiki("Hexanitrobenzene")), "det_velocity": (9340, wiki("Hexanitrobenzene"))}),
    ("Octanitrocubane", "[O-][N+](=O)C12C3([N+]([O-])=O)C4([N+]([O-])=O)C1([N+]([O-])=O)C5([N+]([O-])=O)C2([N+]([O-])=O)C3([N+]([O-])=O)C45[N+]([O-])=O",
     {"density": (1.979, wiki("Octanitrocubane")), "det_velocity": (10100, wiki("Octanitrocubane"))}),
    ("TEX", "C12C3N(C4C(N1[N+](=O)[O-])OC(O4)C(O2)O3)[N+](=O)[O-]",
     {"density": (1.985, wiki("TEX_(explosive)")), "det_velocity": (8500, wiki("TEX_(explosive)"))}),
    ("Nitromethane", "C[N+](=O)[O-]",
     {"density": (1.1371, wiki("Nitromethane", "liquid, 20 C")), "det_velocity": (6400, wiki("Nitromethane")),
      "hf_solid": (-112.6, wiki("Nitromethane", "dHf of the liquid; nitromethane is liquid at 298 K"))}),
    ("2,4-Dinitrotoluene", "Cc1ccc(cc1[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.52, wiki("2,4-Dinitrotoluene", "GESTIS"))}),
    # No value checked against a source: everything predicted. DNAN's Wikipedia
    # density (1.336) looks like a melt value, so it is deliberately not used.
    ("DNAN", "COc1ccc([N+](=O)[O-])cc1[N+](=O)[O-]", {}),
    ("DATB", "Nc1c([N+](=O)[O-])cc([N+](=O)[O-])c(N)c1[N+](=O)[O-]", {}),
]

PUBCHEM_NOTE = "structure checked against PubChem CID {cid}"

# Expansion batches (Oct 2026). Each batch is appended independently; a value is
# 'literature' only when it was read from the cited page's chembox (with its
# references). Structures without a checked value are fully predicted, but their
# SMILES were matched (InChIKey) against the PubChem CID noted in the comment.
BATCH_1_NITRO_AROMATICS = [
    ("1,3-Dinitrobenzene", "C1=CC(=CC(=C1)[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.575, wiki("1,3-Dinitrobenzene"))}),
    ("Picramide", "c1c(cc(c(c1[N+](=O)[O-])N)[N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.8, wiki("2,4,6-Trinitroaniline")), "det_velocity": (7300, wiki("2,4,6-Trinitroaniline"))}),
    ("2,4,6-Trinitroanisole", "COc1c([N+](=O)[O-])cc([N+](=O)[O-])cc1[N+](=O)[O-]",
     {"density": (1.61, wiki("Trinitroanisole"))}),
    ("Styphnic acid", "c1c(c(c(c(c1[N+](=O)[O-])O)[N+](=O)[O-])O)[N+](=O)[O-]",
     {"density": (1.829, wiki("Styphnic_acid"))}),
    ("Picryl chloride", "C1=C(C=C(C(=C1[N+](=O)[O-])Cl)[N+](=O)[O-])[N+](=O)[O-]",
     {"det_velocity": (7200, wiki("Picryl_chloride"))}),
    ("Hexanitrodiphenylamine", "C1=C(C=C(C(=C1[N+](=O)[O-])NC2=C(C=C(C=C2[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
     {"det_velocity": (7100, wiki("Hexanitrodiphenylamine", "D for pressed material; chembox density 1.64 is pressed, not TMD, so not used"))}),
    ("2,4-Dinitrophenol", "O=[N+]([O-])c1cc(ccc1O)[N+]([O-])=O",
     {"det_velocity": (5000, wiki("2,4-Dinitrophenol", "Encyclopedia of Explosives"))}),
    ("2,4-Dinitroaniline", "C1=CC(=C(C=C1[N+](=O)[O-])[N+](=O)[O-])N",
     {"density": (1.61, wiki("2,4-Dinitroaniline"))}),
    ("1,3,5-Triazido-2,4,6-trinitrobenzene", "O=[N+](c1c(N=[N+]=[N-])c([N+]([O-])=O)c(N=[N+]=[N-])c([N+]([O-])=O)c1N=[N+]=[N-])[O-]",
     {"det_velocity": (7350, wiki("1,3,5-Triazido-2,4,6-trinitrobenzene"))}),
    ("Nitrobenzene", "c1ccc(cc1)[N+](=O)[O-]",
     {"density": (1.199, wiki("Nitrobenzene", "liquid"))}),
    ("2,4,6-Trinitrobenzoic acid", "C1=C(C=C(C(=C1[N+](=O)[O-])C(=O)O)[N+](=O)[O-])[N+](=O)[O-]", {}),  # SMILES from Wikipedia chembox
    ("2,6-Dinitrotoluene", "Cc1c([N+](=O)[O-])cccc1[N+](=O)[O-]", {}),  # PubChem CID 11813
    ("1,5-Dinitronaphthalene", "O=[N+]([O-])c1cccc2c([N+](=O)[O-])cccc12", {}),  # CID 11803
    ("Hexanitroazobenzene", "O=[N+]([O-])c1cc([N+](=O)[O-])c(/N=N/c2c([N+](=O)[O-])cc([N+](=O)[O-])cc2[N+](=O)[O-])c([N+](=O)[O-])c1", {}),  # CID 519640
    ("PYX", "O=[N+]([O-])c1cc([N+](=O)[O-])c(Nc2nc(Nc3c([N+](=O)[O-])cc([N+](=O)[O-])cc3[N+](=O)[O-])c([N+](=O)[O-])cc2[N+](=O)[O-])c([N+](=O)[O-])c1", {}),  # CID 14597089
    ("Trinitrophloroglucinol", "O=[N+]([O-])c1c(O)c([N+](=O)[O-])c(O)c([N+](=O)[O-])c1O", {}),  # CID 4555524
    ("Diazodinitrophenol", "[N-]=[N+]=C1C=C([N+](=O)[O-])C=C([N+](=O)[O-])C1=O",
     {"density": (1.715, wiki("Diazodinitrophenol", "midpoint of 1.71-1.72 (pure)"))}),  # CID 5463880
]

BATCH_2_NITRATE_ESTERS_AND_ALIPHATICS = [
    ("Methyl nitrate", "[O-][N+](=O)OC",
     {"density": (1.203, wiki("Methyl_nitrate", "liquid")),
      "det_velocity": (6300, wiki("Methyl_nitrate", "D at rho=1.217, citing Meyer/Koehler/Homburg 'Explosives'"))}),
    ("Ethyl nitrate", "CCO[N+](=O)[O-]", {"density": (1.10, wiki("Ethyl_nitrate", "liquid"))}),
    ("Isopropyl nitrate", "CC(C)O[N+](=O)[O-]", {"density": (1.036, wiki("Isopropyl_nitrate", "liquid"))}),
    ("DEGDN", "C(CO[N+](=O)[O-])OCCO[N+](=O)[O-]",
     {"density": (1.3846, wiki("Diethylene_glycol_dinitrate", "liquid, 20 C"))}),
    ("TEGDN", "C(CO[N+](=O)[O-])OCCOCCO[N+](=O)[O-]", {"density": (1.33, wiki("Triethylene_glycol_dinitrate", "liquid"))}),
    ("TMETN", "CC(CO[N+]([O-])=O)(CO[N+]([O-])=O)CO[N+]([O-])=O", {"density": (1.47, wiki("Trimethylolethane_trinitrate", "liquid"))}),
    ("BTTN", "O=[N+]([O-])OC(CCO[N+]([O-])=O)CO[N+](=O)[O-]", {"density": (1.52, wiki("1,2,4-Butanetriol_trinitrate", "liquid"))}),
    ("PGDN", "O=[N+]([O-])OC(C)CO[N+](=O)[O-]", {"density": (1.232, wiki("Propylene_glycol_dinitrate", "liquid, 20 C, GESTIS"))}),
    ("Nitroethane", "CC[N+](=O)[O-]", {"density": (1.054, wiki("Nitroethane", "liquid"))}),
    ("1-Nitropropane", "CCC[N+](=O)[O-]", {"density": (0.998, wiki("1-Nitropropane", "liquid"))}),
    ("2-Nitropropane", "CC(C)[N+](=O)[O-]", {"density": (0.9821, wiki("2-Nitropropane", "liquid"))}),
    ("Tetranitromethane", "C([N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]",
     {"density": (1.623, wiki("Tetranitromethane", "liquid"))}),
    ("Trinitromethane", "C([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", {"density": (1.469, wiki("Trinitromethane"))}),
    ("Hexanitroethane", "C(C([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-])([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", {}),
    ("2,2,2-Trinitroethanol", "OCC([N+](=O)[O-])([N+](=O)[O-])[N+](=O)[O-]", {}),  # CID 61233
]

BATCH_3_NITRAMINES_AND_UREAS = [
    ("DINGU", "C12C(NC(=O)N1[N+](=O)[O-])N(C(=O)N2)[N+](=O)[O-]",
     {"density": (1.94, wiki("Dinitroglycoluril", "Zhao et al.")),
      "det_velocity": (8450, wiki("Dinitroglycoluril", "Zhao et al.")),
      "hf_solid": (-74 * 4.184, wiki("Dinitroglycoluril", "dHf -74 kcal/mol (Khire), converted to kJ/mol"))}),
    ("EDNA", "C(CN[N+](=O)[O-])N[N+](=O)[O-]", {"density": (1.71, wiki("Ethylenedinitramine"))}),
    ("Nitrourea", "C(=O)(N)N[N+](=O)[O-]",
     {"density": (1.73, wiki("Nitrourea", "chembox D is uncited, so not used"))}),
    ("Keto-RDX", "O=C1N([N+](=O)[O-])CN([N+](=O)[O-])CN1[N+](=O)[O-]", {}),  # K-6, CID 10933462
    ("BCHMX", "O=[N+]([O-])N1CN([N+](=O)[O-])C2C1N([N+](=O)[O-])CN2[N+](=O)[O-]", {}),  # CID 10870106
    ("Heptanitrocubane", "[O-][N+](=O)C14C3C2([N+]([O-])=O)C1([N+]([O-])=O)C5([N+]([O-])=O)C2([N+]([O-])=O)C3([N+]([O-])=O)C45[N+]([O-])=O", {}),
]

BATCH_4_HETEROCYCLES = [
    ("ANPZ", "Nc1nc(N)c([N+](=O)[O-])nc1[N+](=O)[O-]", {}),  # CID 4326084
    ("LLM-116", "Nc1c([N+](=O)[O-])n[nH]c1[N+](=O)[O-]", {}),  # CID 11480728
    ("ANTA", "Nc1n[nH]c([N+](=O)[O-])n1", {}),  # CID 135409704
    ("DNTF", "O=[N+]([O-])c1nonc1-c1no[n+]([O-])c1-c1nonc1[N+](=O)[O-]", {}),  # CID 605902
    ("DAAF", "Nc1nonc1/N=[N+](/[O-])c1nonc1N", {}),  # CID 533871
    ("3,4-Dinitropyrazole", "O=[N+]([O-])c1c[nH]nc1[N+](=O)[O-]", {}),  # CID 3620736
    ("3,5-Dinitropyrazole", "O=[N+]([O-])c1cc([N+](=O)[O-])n[nH]1", {}),  # CID 142323
    ("2,4-Dinitroimidazole", "O=[N+]([O-])c1c[nH]c([N+](=O)[O-])n1", {}),  # CID 21275
    ("Trinitrotriazine", "O=[N+]([O-])C1=NC([N+]([O-])=O)=NC([N+]([O-])=O)=N1", {}),
    ("Cyanuric triazide", "C1(=NC(=NC(=N1)N=[N+]=[N-])N=[N+]=[N-])N=[N+]=[N-]",
     {"density": (1.73, wiki("Cyanuric_triazide", "CRC Handbook 93rd ed."))}),
]

BATCH_5_PEROXIDES = [
    ("TATP", "CC1(C)OOC(C)(C)OOC(C)(C)OO1",
     {"det_velocity": (5300, wiki("Acetone_peroxide", "D at maximum density 1.18 g/cm3"))}),  # CID 4380970
    ("HMTD", "C1N2COOCN(COO1)COOC2",
     {"density": (1.57, wiki("Hexamethylene_triperoxide_diamine"))}),
]

EXPANSION_BATCHES = [BATCH_1_NITRO_AROMATICS, BATCH_2_NITRATE_ESTERS_AND_ALIPHATICS,
                     BATCH_3_NITRAMINES_AND_UREAS, BATCH_4_HETEROCYCLES, BATCH_5_PEROXIDES]

PRED_NOTE = "XGBoost prediction (models/*.joblib, descriptor contract v1); no checked literature value"


def main() -> None:
    predictor = PropertyPredictor(ROOT / "models")
    rows, seen = [], set()
    entries = [(n, s, l, True) for n, s, l in MOLECULES]
    entries += [(n, s, l, False) for batch in EXPANSION_BATCHES for n, s, l in batch]
    for name, smi, lit, core in entries:
        mol = Chem.MolFromSmiles(smi)
        bad = mol is None or "." in smi or Chem.GetFormalCharge(mol) != 0 \
            or any(a.GetFormalCharge() and not a.GetNeighbors() for a in mol.GetAtoms())
        can = None if mol is None else Chem.MolToSmiles(mol)
        if bad or can in seen:
            if core:
                raise ValueError(f"{name}: invalid, charged/salt, or duplicate")
            print(f"skipping {name}: invalid, charged/salt, or duplicate")
            continue
        seen.add(can)
        preds = predictor.predict(can) or {}
        row = {"name": name, "SMILES": can}
        for key in PROPERTY_KEYS:
            if key in lit:
                val, cite = lit[key]
                if isinstance(val, tuple):  # kcal/kg -> kJ/mol
                    val = val[1] * 4.184 * MolWt(mol) / 1000.0
                row[key] = round(float(val), 4)
                row[f"{key}_source"] = cite
                row[f"{key}_provenance"] = "literature"
            else:
                row[key] = round(float(preds[key].value), 4)
                row[f"{key}_source"] = PRED_NOTE
                row[f"{key}_provenance"] = "predicted"
        rows.append(row)
    cols = ["name", "SMILES"] + [c for k in PROPERTY_KEYS for c in (k, f"{k}_source", f"{k}_provenance")]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} molecules to {OUT}")


if __name__ == "__main__":
    main()
