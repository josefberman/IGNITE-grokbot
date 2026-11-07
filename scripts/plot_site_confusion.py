#!/usr/bin/env python
"""Site confusion matrices on the evaluation split.

Each labeled reagent has a list of applicable sites: the distinct parent atom
sets that yield a valid product. Sites are numbered from 1 in atom-index order,
so within one parent and reagent the same number is the same atoms.

The row is the site stored on the pair. The planner column is the applicable
site with the highest mean site-head score on its reacting atoms. The
rule-based column is the first product in a seeded shuffle of that reagent's
products. Both choices use the labeled reagent, so this is the site decision
rather than the reagent decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROP_COLS = ("density", "det_velocity", "det_pressure", "hf_solid")
RULE_SEED = 0


def plot_confusion(true_idx, pred_idx, labels: list[str], out_dir: Path, stem: str, title: str) -> None:
    """Rows are the true site, columns the selected site. Color is the row fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n_classes = len(labels)
    counts = np.zeros((n_classes, n_classes), dtype=float)
    for truth, pred in zip(true_idx, pred_idx):
        counts[int(truth), int(pred)] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    fraction = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)

    side = max(6.0, n_classes * 0.36)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(side + 1.6, side))
    image = ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues", origin="upper", aspect="equal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true site")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(axis="both", length=2, pad=1)
    ax.set_xlabel("Selected site")
    ax.set_ylabel("True site")
    ax.set_title(title)
    fig.tight_layout()
    for ext, kw in (("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(out_dir / f"{stem}.{ext}", **kw)
    plt.close(fig)
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true_site", "selected_site", "count", "row_fraction"])
        for i in range(n_classes):
            for j in range(n_classes):
                if counts[i, j] == 0:
                    continue
                w.writerow([labels[i], labels[j], int(counts[i, j]), f"{fraction[i, j]:.6g}"])


def _reaction(smarts: str, cache: dict):
    if smarts in cache:
        return cache[smarts]
    from rdkit.Chem import AllChem

    try:
        rxn = AllChem.ReactionFromSmarts(smarts)
    except Exception:
        rxn = None
    cache[smarts] = rxn
    return rxn


def applicable_sites(mol, smiles: str, smarts_list: list[str], rxn_cache: dict) -> list[tuple]:
    """Valid products as (child_smiles, reacting_atoms, match_atoms), in encounter order."""
    from rdkit import Chem

    from ml.data_gen.generate_pairs import reacting_atom_indices

    raw = []
    for smarts in smarts_list:
        rxn = _reaction(smarts, rxn_cache)
        if rxn is None:
            continue
        try:
            products = rxn.RunReactants((mol,))
        except Exception:
            continue
        try:
            tmpl = rxn.GetReactantTemplate(0)
            matches = mol.GetSubstructMatches(tmpl) if tmpl is not None else ()
        except Exception:
            matches = ()
        for pi, product_tuple in enumerate(products):
            reacting = tuple(int(a) for a in reacting_atom_indices(mol, rxn, pi))
            match = tuple(int(a) for a in matches[pi]) if pi < len(matches) else reacting
            for product in product_tuple:
                try:
                    Chem.SanitizeMol(product)
                    child = Chem.MolToSmiles(product, canonical=True)
                except Exception:
                    continue
                if not child or "." in child or child == smiles:
                    continue
                raw.append((child, reacting, match))
                break
    return raw


def rule_choice(raw: list[tuple]) -> tuple | None:
    """First product after the seeded shuffle used by the rule-based modifier."""
    seen: dict[str, tuple] = {}
    for child, reacting, _match in raw:
        if child not in seen and reacting:
            seen[child] = reacting
    if not seen:
        return None
    items = sorted(seen.items())
    random.Random(RULE_SEED).shuffle(items)
    return items[0][1]


def encode_parent(model, atom_x, edge):
    """Gap-independent encoder state. Message passing does not see the gap."""
    import torch

    with torch.inference_mode():
        parent_z, atom_ctx, atom_local = model.parent_enc(atom_x, edge, return_atom_emb=True)
    return parent_z, atom_ctx, atom_local


def site_scores(model, encoded, gap):
    """Match ModificationPlanner.encode, then the site head, for one cached parent."""
    import torch

    parent_z, atom_ctx, atom_local = encoded
    with torch.inference_mode():
        gamma, beta = model.film(gap).chunk(2, dim=-1)
        parent_cond = parent_z * (1.0 + gamma) + beta
        n_atoms = atom_local.size(1)
        gap_exp = gap.unsqueeze(1).expand(-1, n_atoms, -1)
        attn = model.atom_score(torch.cat([atom_local, gap_exp], dim=-1)).squeeze(-1)
        alpha = torch.softmax(attn, dim=-1)
        attended = torch.bmm(alpha.unsqueeze(1), atom_local).squeeze(1)
        z = model.fuse(torch.cat([parent_cond, attended, gap], dim=-1))
        atom_emb = torch.cat([atom_local, atom_ctx], dim=-1)
        return model.forward_sites(atom_emb, z, gap)


def planner_choice(raw: list[tuple], scores) -> tuple | None:
    """Applicable atom set with the highest mean site-head score."""
    values = [float(x) for x in scores]
    best: tuple | None = None
    best_key: tuple[float, int] | None = None
    seen: set[tuple] = set()
    for _child, reacting, _match in raw:
        if not reacting or reacting in seen:
            continue
        seen.add(reacting)
        picked = [values[atom] for atom in reacting if 0 <= atom < len(values)]
        if not picked:
            continue
        key = (sum(picked) / len(picked), -min(reacting))
        if best_key is None or key > best_key:
            best_key = key
            best = reacting
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "modification_pairs.csv")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "ml" / "checkpoints")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    args = ap.parse_args(argv)

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    import pandas as pd
    import torch

    from backend.app.services.modification import LearnedPlannerBundle
    from backend.app.services.reaction_primitives import list_primitives
    from ml.models.architectures import mol_to_graph
    from ml.planner_gap import planner_input_from_parent_target

    if not args.data.exists():
        print(f"missing {args.data}")
        return 1
    bundle = LearnedPlannerBundle(args.checkpoint_dir)
    if not bundle.available or bundle._model is None:
        print(bundle.unavailable_reason)
        return 1

    smarts_by_reagent = {
        p["family_id"]: list(p.get("smarts_variants") or [])
        for p in list_primitives()
        if p["family_id"] != "identity"
    }
    usecols = ["split", "parent_smiles", "reagent_id", "site_atom_indices"]
    usecols += [f"parent_{k}" for k in PROP_COLS] + [f"child_{k}" for k in PROP_COLS]
    df = pd.read_csv(args.data, usecols=usecols)
    df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        print(f"no rows in split {args.split}")
        return 1

    model = bundle._model
    rxn_cache: dict = {}
    mol_cache: dict = {}
    site_cache: dict = {}
    encoded_cache: dict = {}
    rows: list[tuple[int, int, int]] = []
    n = len(df)
    n_unlabeled = n_no_site = n_label_miss = 0

    for i, rec in enumerate(df.itertuples(index=False), start=1):
        try:
            label = tuple(int(a) for a in json.loads(rec.site_atom_indices))
        except (TypeError, json.JSONDecodeError):
            label = tuple()
        if not label:
            n_unlabeled += 1
            continue
        smiles = str(rec.parent_smiles)
        reagent = str(rec.reagent_id)
        key = (smiles, reagent)
        packed = site_cache.get(key)
        if packed is None:
            mol = mol_cache.get(smiles)
            if smiles not in mol_cache:
                mol = Chem.MolFromSmiles(smiles)
                mol_cache[smiles] = mol
            raw = []
            if mol is not None:
                raw = applicable_sites(mol, smiles, smarts_by_reagent.get(reagent, []), rxn_cache)
            classes = sorted({reacting for _child, reacting, _match in raw if reacting})
            rule = rule_choice(raw)
            rule_index = classes.index(rule) if rule in classes else None
            packed = (raw, classes, rule_index)
            site_cache[key] = packed
        raw, classes, rule_index = packed
        if not classes or label not in classes or rule_index is None:
            if not classes or rule_index is None:
                n_no_site += 1
            else:
                n_label_miss += 1
            continue

        if smiles not in encoded_cache:
            graph = mol_to_graph(smiles)
            if graph is None:
                encoded_cache[smiles] = None
            else:
                x, edge = graph
                encoded_cache[smiles] = encode_parent(model, x.unsqueeze(0), edge)
        encoded = encoded_cache[smiles]
        if encoded is None:
            n_no_site += 1
            continue
        parent = {k: float(getattr(rec, f"parent_{k}")) for k in PROP_COLS}
        target = {k: float(getattr(rec, f"child_{k}")) for k in PROP_COLS}
        gap = torch.tensor([planner_input_from_parent_target(parent, target)], dtype=torch.float32)
        chosen = planner_choice(raw, site_scores(model, encoded, gap))
        if chosen not in classes:
            n_no_site += 1
            continue
        rows.append((classes.index(label), classes.index(chosen), rule_index))
        if i % 2000 == 0 or i == n:
            print(f"  {i}/{n} kept={len(rows)}", flush=True)

    if not rows:
        print("no comparable rows")
        return 1

    n_classes = max(max(t, p, r) for t, p, r in rows) + 1
    labels = [f"site {i + 1}" for i in range(n_classes)]
    true_idx = [t for t, _p, _r in rows]
    planner_idx = [p for _t, p, _r in rows]
    rule_idx = [r for _t, _p, r in rows]
    plot_confusion(
        true_idx, planner_idx, labels, args.out_dir,
        "site_confusion_planner", "Planner site confusion (evaluation set)",
    )
    plot_confusion(
        true_idx, rule_idx, labels, args.out_dir,
        "site_confusion_rule_based", "Rule-based site confusion (evaluation set)",
    )
    kept = len(rows)
    planner_hit = sum(t == p for t, p, _r in rows) / kept
    rule_hit = sum(t == r for t, _p, r in rows) / kept
    print(
        f"n={kept} unlabeled={n_unlabeled} no_site={n_no_site} label_miss={n_label_miss} "
        f"planner_site={planner_hit:.3f} rule_site={rule_hit:.3f}"
    )
    for name in ("site_confusion_planner", "site_confusion_rule_based"):
        print(f"wrote {args.out_dir / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
