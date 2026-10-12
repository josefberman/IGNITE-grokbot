"""Tests for SMARTS reagent IDs and reacting-atom bookkeeping (no torch)."""

from backend.app.services.reaction_primitives import list_planner_vocab, list_primitives, list_reagents


def test_list_reagents_stable_ids():
    reagents = list_reagents()
    assert len(reagents) > len([p for p in list_primitives() if p["family_id"] != "identity"])
    ids = [r["reagent_id"] for r in reagents]
    assert len(ids) == len(set(ids))
    assert all("__v" in rid for rid in ids)
    groups, reagent_rows = list_planner_vocab()
    assert groups
    assert reagent_rows


def test_reacting_atom_indices_nonzero():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from ml.data_gen.generate_pairs import reacting_atom_indices

    mol = Chem.MolFromSmiles("c1ccccc1")
    rxn = AllChem.ReactionFromSmarts("[cH:1]>>[c:1]N")
    assert rxn is not None
    sites = reacting_atom_indices(mol, rxn, 0)
    assert len(sites) >= 1
    assert all(0 <= i < mol.GetNumAtoms() for i in sites)
