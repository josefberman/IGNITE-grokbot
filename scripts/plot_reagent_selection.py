#!/usr/bin/env python
"""Scatter selected reagents on the evaluation split.

Each reagent id gets a stable integer (sorted id order). For every test pair
the x coordinate is the reagent the rule-based ranker selects, and the y
coordinate is the reagent the learned planner selects. A hollow circle at
(label, label) marks the true reagent.

The gap shown to both models is the child properties minus the parent
properties, the same target the planner is trained to recover. The rule-based
choice is the first gap-ranked reagent that matches the parent. The planner
choice is its top family, then the top reagent inside that family.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROP_COLS = ("density", "det_velocity", "det_pressure", "hf_solid")


def reagent_codes(ids: list[str]) -> dict[str, int]:
    return {rid: i for i, rid in enumerate(sorted(set(ids)))}


def rule_selected(smiles: str, parent: dict, target: dict, applicable: set[str]) -> str | None:
    from backend.app.services.reagents import rank_reagents_for_gap

    ranked = rank_reagents_for_gap(parent, target)
    if applicable:
        for rid in ranked:
            if rid in applicable:
                return rid
    return ranked[0] if ranked else None


def planner_selected(
    bundle,
    smiles: str,
    gap: list[float],
    groups: set[str],
    reactions: set[str],
    graphs: dict,
) -> str | None:
    import torch

    from ml.models.architectures import mol_to_graph

    if smiles not in graphs:
        graphs[smiles] = mol_to_graph(smiles)
    graph = graphs[smiles]
    if graph is None or bundle._model is None:
        return None
    x, edge = graph
    if x.numel() == 0:
        return None
    model = bundle._model
    gap_t = torch.tensor([gap], dtype=torch.float32)
    with torch.no_grad():
        z, _, atom_emb = model.encode(x.unsqueeze(0), gap_t, edge)
        fam_logits = model.forward_family(z, gap_t).squeeze(0).clone()
        if groups:
            for i, name in enumerate(bundle.family_names):
                if name not in groups:
                    fam_logits[i] = -1e9
            if not bool((fam_logits > -1e8).any()):
                fam_logits = model.forward_family(z, gap_t).squeeze(0)
        if not bool((fam_logits > -1e8).any()):
            return None
        fam_i = int(fam_logits.argmax())
        allowed = set(bundle.family_to_reagent_ids.get(bundle.family_names[fam_i], []))
        reag_logits = model.forward_reagent(
            z, family_idx=torch.tensor([fam_i]), gap=gap_t, atom_emb=atom_emb
        ).squeeze(0).clone()
        for i, rid in enumerate(bundle.reagent_names):
            if rid not in allowed or (reactions and rid not in reactions):
                reag_logits[i] = -1e9
        if reactions and not bool((reag_logits > -1e8).any()):
            reag_logits = model.forward_reagent(
                z, family_idx=torch.tensor([fam_i]), gap=gap_t, atom_emb=atom_emb
            ).squeeze(0).clone()
            for i, rid in enumerate(bundle.reagent_names):
                if rid not in allowed:
                    reag_logits[i] = -1e9
        if not bool((reag_logits > -1e8).any()):
            return None
        return bundle.reagent_names[int(reag_logits.argmax())]


def plot(rule_idx, planner_idx, true_idx, out_dir: Path, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    rng = np.random.default_rng(0)
    jitter = 0.18
    x = np.asarray(rule_idx, dtype=float) + rng.uniform(-jitter, jitter, len(rule_idx))
    y = np.asarray(planner_idx, dtype=float) + rng.uniform(-jitter, jitter, len(planner_idx))
    true = np.asarray(true_idx, dtype=float)
    hi = int(max(rule_idx + planner_idx + true_idx))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(
        true, true, s=42, facecolors="none", edgecolors="#D55E00",
        linewidths=1.1, label="True reagent", zorder=3,
    )
    ax.scatter(
        x, y, s=14, c="#0072B2", alpha=0.35, linewidths=0,
        label="Rule-based (x), planner (y)", zorder=2,
    )
    ax.plot([-0.5, hi + 0.5], [-0.5, hi + 0.5], color="0.6", lw=0.8, zorder=1)
    ax.set_xlim(-0.5, hi + 0.5)
    ax.set_ylim(-0.5, hi + 0.5)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.tick_params(axis="both", labelsize=5)
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.set_xlabel("Reagent selected by the rule-based model")
    ax.set_ylabel("Reagent selected by the planner")
    ax.set_title("Selected reagents on the evaluation set")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(out_dir / f"{stem}.{ext}", **kw)
    plt.close(fig)


def plot_confusion(
    true_idx, pred_idx, n_classes: int, out_dir: Path, stem: str, title: str,
    labels: list[str] | None = None,
) -> None:
    """Rows are the true reagent, columns the selected reagent. Color is the row fraction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    counts = np.zeros((n_classes, n_classes), dtype=float)
    for truth, pred in zip(true_idx, pred_idx):
        counts[int(truth), int(pred)] += 1
    row_sum = counts.sum(axis=1, keepdims=True)
    fraction = np.divide(counts, row_sum, out=np.zeros_like(counts), where=row_sum > 0)

    names = labels if labels is not None and len(labels) == n_classes else [str(i) for i in range(n_classes)]
    side = max(8.0, n_classes * 0.22)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(side + 1.8, side))
    image = ax.imshow(fraction, vmin=0, vmax=1, cmap="Blues", origin="upper", aspect="equal")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Fraction of true reagent")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(names, rotation=90, ha="center", fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.tick_params(axis="both", length=2, pad=1)
    ax.set_xlabel("Selected reagent")
    ax.set_ylabel("True reagent")
    ax.set_title(title)
    fig.tight_layout()
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 200})):
        fig.savefig(out_dir / f"{stem}.{ext}", **kw)
    plt.close(fig)
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["true_index", "selected_index", "count", "row_fraction"])
        for i in range(n_classes):
            for j in range(n_classes):
                if counts[i, j] == 0:
                    continue
                w.writerow([i, j, int(counts[i, j]), f"{fraction[i, j]:.6g}"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "modification_pairs.csv")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "ml" / "checkpoints")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures")
    args = ap.parse_args(argv)

    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    import pandas as pd

    from backend.app.services.modification import LearnedPlannerBundle
    from backend.app.services.reaction_primitives import matching_group_ids, matching_reaction_ids
    from backend.app.services.reagents import REAGENTS
    from ml.planner_gap import planner_input_from_parent_target

    if not args.data.exists():
        print(f"missing {args.data}")
        return 1
    bundle = LearnedPlannerBundle(args.checkpoint_dir)
    if not bundle.available:
        print(bundle.unavailable_reason)
        return 1

    usecols = ["split", "parent_smiles", "reagent_id", "reaction_family"]
    usecols += [f"parent_{k}" for k in PROP_COLS] + [f"child_{k}" for k in PROP_COLS]
    df = pd.read_csv(args.data, usecols=usecols)
    df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)
    if df.empty:
        print(f"no rows in split {args.split}")
        return 1

    codes = reagent_codes(
        [r.reagent_id for r in REAGENTS] + list(bundle.reagent_names) + df["reagent_id"].astype(str).tolist()
    )
    code_path = args.out_dir / "reagent_index.csv"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with code_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "reagent_id"])
        for rid, idx in sorted(codes.items(), key=lambda kv: kv[1]):
            w.writerow([idx, rid])

    match_cache: dict[str, tuple[set[str], set[str]]] = {}
    graphs: dict = {}

    def applicability(smiles: str) -> tuple[set[str], set[str]]:
        hit = match_cache.get(smiles)
        if hit is None:
            hit = (set(matching_group_ids(smiles)), set(matching_reaction_ids(smiles)))
            match_cache[smiles] = hit
        return hit

    rows = []
    n = len(df)
    for i, rec in enumerate(df.itertuples(index=False), start=1):
        smiles = str(rec.parent_smiles)
        parent = {k: float(getattr(rec, f"parent_{k}")) for k in PROP_COLS}
        target = {k: float(getattr(rec, f"child_{k}")) for k in PROP_COLS}
        groups, reactions = applicability(smiles)
        rule = rule_selected(smiles, parent, target, reactions)
        planner = planner_selected(
            bundle, smiles, planner_input_from_parent_target(parent, target), groups, reactions, graphs
        )
        truth = str(rec.reagent_id)
        if rule is None or planner is None or truth not in codes or rule not in codes or planner not in codes:
            continue
        rows.append((codes[rule], codes[planner], codes[truth], rule, planner, truth))
        if i % 1000 == 0 or i == n:
            print(f"  {i}/{n} kept={len(rows)}", flush=True)

    if not rows:
        print("no comparable rows")
        return 1

    rule_idx = [r[0] for r in rows]
    planner_idx = [r[1] for r in rows]
    true_idx = [r[2] for r in rows]
    stem = "reagent_selection_eval"
    plot(rule_idx, planner_idx, true_idx, args.out_dir, stem)
    n_classes = max(codes.values()) + 1
    names = [""] * n_classes
    for rid, idx in codes.items():
        names[idx] = rid
    plot_confusion(
        true_idx, planner_idx, n_classes, args.out_dir,
        "reagent_confusion_planner", "Planner reagent confusion (evaluation set)",
        labels=names,
    )
    plot_confusion(
        true_idx, rule_idx, n_classes, args.out_dir,
        "reagent_confusion_rule_based", "Rule-based reagent confusion (evaluation set)",
        labels=names,
    )
    points = args.out_dir / f"{stem}.csv"
    with points.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rule_index", "planner_index", "true_index", "rule_reagent", "planner_reagent", "true_reagent"])
        w.writerows(rows)
    both = sum(a == b == c for a, b, c in zip(rule_idx, planner_idx, true_idx))
    print(
        f"n={len(rows)} planner_correct={sum(a == c for a, c in zip(planner_idx, true_idx)) / len(rows):.3f} "
        f"rule_correct={sum(a == c for a, c in zip(rule_idx, true_idx)) / len(rows):.3f} "
        f"both_correct={both / len(rows):.3f}"
    )
    print(f"wrote {code_path}")
    for name in (
        stem,
        "reagent_confusion_planner",
        "reagent_confusion_rule_based",
    ):
        print(f"wrote {args.out_dir / (name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
