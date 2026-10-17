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

PRED_NOTE = "XGBoost prediction (models/*.joblib, descriptor contract v1); no checked literature value"


def main() -> None:
    predictor = PropertyPredictor(ROOT / "models")
    rows, seen = [], set()
    for name, smi, lit in MOLECULES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"invalid SMILES for {name}")
        if Chem.GetFormalCharge(mol) != 0 or "." in smi:
            raise ValueError(f"{name} is charged or a salt")
        can = Chem.MolToSmiles(mol)
        if can in seen:
            raise ValueError(f"duplicate {name}")
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
