"""Simple graph / gap encoders and hierarchical modification planner (PyTorch)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem


ATOM_LIST = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
# H, B, C, N, O, F, Si, P, S, Cl, Br, I

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

RING_SIZES = (3, 4, 5, 6, 7, 8)

# element one-hot (+ unknown) + hybridization (+ other) + scalars + ring-size flags
ATOM_FEAT_DIM = (
    len(ATOM_LIST)
    + 1
    + len(HYBRIDIZATION_LIST)
    + 1
    + 12
    + len(RING_SIZES)
)


def _atom_features(atom: Chem.Atom) -> List[float]:
    z = atom.GetAtomicNum()
    feats: List[float] = [1.0 if z == a else 0.0 for a in ATOM_LIST]
    feats.append(0.0 if z in ATOM_LIST else 1.0)

    hyb = atom.GetHybridization()
    feats.extend(1.0 if hyb == h else 0.0 for h in HYBRIDIZATION_LIST)
    feats.append(0.0 if hyb in HYBRIDIZATION_LIST else 1.0)

    # rdkit-pypi 2022.x uses GetExplicitValence/GetImplicitValence;
    # RDKit 2024+ renamed those to GetValence(ValenceType.*).
    if hasattr(atom, "GetValence") and hasattr(Chem, "ValenceType"):
        explicit_v = atom.GetValence(Chem.ValenceType.EXPLICIT)
        implicit_v = atom.GetValence(Chem.ValenceType.IMPLICIT)
    else:
        explicit_v = atom.GetExplicitValence()
        implicit_v = atom.GetImplicitValence()
    feats.extend(
        [
            float(atom.GetTotalDegree()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalValence()),
            float(explicit_v),
            float(implicit_v),
            float(atom.GetTotalNumHs()),
            float(atom.GetNumRadicalElectrons()),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            float(atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED),
            float(atom.GetMass()) / 100.0,
            float(z) / 100.0,
        ]
    )
    feats.extend(float(atom.IsInRingSize(s)) for s in RING_SIZES)
    assert len(feats) == ATOM_FEAT_DIM
    return feats


BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
# Listed bond types plus "other".
N_BOND_TYPES = len(BOND_TYPE_LIST) + 1


def _bond_type_index(bond: Chem.Bond) -> int:
    bt = bond.GetBondType()
    return BOND_TYPE_LIST.index(bt) if bt in BOND_TYPE_LIST else len(BOND_TYPE_LIST)


def mol_to_graph(smiles: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Atom features and a ``[3, E]`` edge tensor: source, target, bond-type index.

    Encoders without bond embeddings read only the first two rows.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats = [_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float32)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _bond_type_index(bond)
        edges.append([i, j, bt])
        edges.append([j, i, bt])
    if not edges:
        edge_index = torch.zeros((3, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


DEFAULT_HIDDEN = 128
DEFAULT_LAYERS = 3
DEFAULT_BOND_TYPES = N_BOND_TYPES
DEFAULT_SITE_REAGENT = True
DEFAULT_HEADS = 4  # retained for CLI / checkpoint meta compatibility; unused by GraphEncoder
DEFAULT_DROPOUT = 0.15
# Hidden layers before the logits. 0 is a single linear map.
DEFAULT_FAMILY_LAYERS = 1
DEFAULT_REAGENT_LAYERS = 1
DEFAULT_SITE_LAYERS = 1


def _mlp_head(in_dim: int, hidden: int, out_dim: int, n_hidden: int, dropout: float) -> nn.Module:
    """``n_hidden`` hidden layers, then a linear map to ``out_dim``. ``0`` is one linear layer."""
    if n_hidden <= 0:
        return nn.Linear(in_dim, out_dim)
    mods: List[nn.Module] = []
    width = in_dim
    for _ in range(n_hidden):
        mods.extend((nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout)))
        width = hidden
    mods.append(nn.Linear(width, out_dim))
    return nn.Sequential(*mods)


def _site_mix(in_dim: int, hidden: int, n_hidden: int) -> nn.Module:
    """Hidden stack for the site head. At least one layer; no dropout, matching the shipped head."""
    if n_hidden < 1:
        raise ValueError("site head needs at least one hidden layer")
    mods: List[nn.Module] = []
    width = in_dim
    for _ in range(n_hidden):
        mods.extend((nn.Linear(width, hidden), nn.GELU()))
        width = hidden
    return nn.Sequential(*mods)


class BondMessagePass(nn.Module):
    """One residual round of bond-neighbor mean aggregation.

    With ``bond_types > 0`` each message adds an embedding of its bond type;
    ``0`` keeps the untyped messages of older checkpoints.
    """

    def __init__(self, hidden: int, bond_types: int = 0):
        super().__init__()
        self.self_lin = nn.Linear(hidden, hidden)
        self.nei_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.bond_emb = nn.Embedding(bond_types, hidden) if bond_types > 0 else None

    def forward(self, h: torch.Tensor, edge_index: Optional[torch.Tensor]) -> torch.Tensor:
        # h: [B, N, H]. Residual keeps the atom's own identity for the site head.
        if edge_index is None or edge_index.numel() == 0:
            return h + self.norm(F.gelu(self.self_lin(h)))
        B, N, H = h.shape
        if B != 1:
            raise ValueError("BondMessagePass currently supports batch size 1")
        h0 = h.squeeze(0)
        src, dst = edge_index[0], edge_index[1]
        agg = torch.zeros(N, H, device=h.device, dtype=h.dtype)
        deg = torch.zeros(N, 1, device=h.device, dtype=h.dtype)
        msg = h0[src]
        if self.bond_emb is not None and edge_index.size(0) > 2:
            msg = msg + self.bond_emb(edge_index[2])
        agg.index_add_(0, dst, msg)
        deg.index_add_(0, dst, torch.ones(len(dst), 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        mixed = self.self_lin(h0) + self.nei_lin(agg)
        return (h0 + self.norm(F.gelu(mixed))).unsqueeze(0)


class GraphEncoder(nn.Module):
    """Atom projection, residual bond message passing, mean+max readout."""

    def __init__(
        self,
        in_dim: int = ATOM_FEAT_DIM,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,  # unused; kept for call-site compatibility
        dropout: float = DEFAULT_DROPOUT,
        bond_types: int = DEFAULT_BOND_TYPES,
    ):
        super().__init__()
        _ = heads  # encoder no longer uses multi-head attention
        self.hidden = hidden
        self.input = nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden))
        self.mp_layers = nn.ModuleList(
            [BondMessagePass(hidden, bond_types) for _ in range(max(1, layers))]
        )
        self.dropout = nn.Dropout(dropout)
        self.readout = nn.Linear(hidden * 2, hidden)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        *,
        return_atom_emb: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: [B, N, F]
        local = self.input(x)
        h = local
        for mp in self.mp_layers:
            h = self.dropout(mp(h, edge_index))
        atom_emb = h
        if mask is not None:
            keep = (~mask).unsqueeze(-1).to(dtype=h.dtype)
            denom = keep.sum(dim=1).clamp_min(1.0)
            mean = (h * keep).sum(dim=1) / denom
            masked = h.masked_fill(~keep.bool(), -1e9)
            mx = masked.max(dim=1).values
        else:
            mean = h.mean(dim=1)
            mx = h.max(dim=1).values
        graph_emb = self.readout(torch.cat([mean, mx], dim=-1))
        if return_atom_emb:
            return graph_emb, atom_emb, local
        return graph_emb


# Backward-compatible alias
GraphTransformerEncoder = GraphEncoder


class ModificationPlanner(nn.Module):
    """Hierarchical: family → site → reagent (family-conditioned).

    FiLM on the graph vector, atom attention + fuse, and heads that also see the
    raw 4-d gap. ``bond_types=0``, ``site_reagent=False``, and
    ``family_layers=0`` rebuild checkpoints written before typed bonds and the
    reagent-conditioned site head.
    """

    def __init__(
        self,
        n_families: int,
        n_reagents: int,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        heads: int = DEFAULT_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        gap_dim: int = 4,
        family_layers: int = DEFAULT_FAMILY_LAYERS,
        reagent_layers: int = DEFAULT_REAGENT_LAYERS,
        site_layers: int = DEFAULT_SITE_LAYERS,
        bond_types: int = DEFAULT_BOND_TYPES,
        site_reagent: bool = DEFAULT_SITE_REAGENT,
    ):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        self.n_families = n_families
        self.n_reagents = n_reagents
        self.gap_dim = gap_dim
        self.family_layers = family_layers
        self.reagent_layers = reagent_layers
        self.site_layers = site_layers
        self.bond_types = bond_types
        self.site_reagent = site_reagent
        enc_kw = dict(hidden=hidden, layers=layers, heads=heads, dropout=dropout, bond_types=bond_types)
        self.parent_enc = GraphEncoder(**enc_kw)
        self.film = nn.Linear(gap_dim, hidden * 2)
        self.atom_score = nn.Linear(hidden + gap_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + gap_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.family_head = _mlp_head(hidden + gap_dim, hidden, n_families, family_layers, dropout)
        self.reagent_head = _mlp_head(
            hidden + n_families + gap_dim + hidden * 2,
            hidden,
            n_reagents,
            reagent_layers,
            dropout,
        )
        site_in = hidden * 2 + hidden + gap_dim
        if site_reagent:
            self.site_reagent_emb = nn.Embedding(n_reagents, hidden)
            site_in += hidden
        else:
            self.site_reagent_emb = None
        self.site_mix = _site_mix(site_in, hidden, site_layers)
        self.site_head = nn.Linear(hidden, 1)

    def arch_meta(self) -> dict:
        """Constructor settings a checkpoint needs to rebuild this model."""
        return {
            "hidden": self.hidden,
            "layers": self.layers,
            "heads": self.heads,
            "dropout": self.dropout,
            "family_layers": self.family_layers,
            "reagent_layers": self.reagent_layers,
            "site_layers": self.site_layers,
            "bond_types": self.bond_types,
            "site_reagent": self.site_reagent,
        }

    def encode(
        self,
        atom_x: torch.Tensor,
        delta: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # atom_x: [1, N, F], delta: [1, 4] -> z [1, H], parent_z [1, H], atom_emb [1, N, 2H]
        parent_z, atom_ctx, atom_local = self.parent_enc(atom_x, edge_index, return_atom_emb=True)
        gamma, beta = self.film(delta).chunk(2, dim=-1)
        parent_cond = parent_z * (1.0 + gamma) + beta
        n_atoms = atom_local.size(1)
        gap_exp = delta.unsqueeze(1).expand(-1, n_atoms, -1)
        attn = self.atom_score(torch.cat([atom_local, gap_exp], dim=-1)).squeeze(-1)
        alpha = torch.softmax(attn, dim=-1)
        attended = torch.bmm(alpha.unsqueeze(1), atom_local).squeeze(1)
        z = self.fuse(torch.cat([parent_cond, attended, delta], dim=-1))
        atom_emb = torch.cat([atom_local, atom_ctx], dim=-1)
        return z, parent_z, atom_emb

    def _gap(self, z: torch.Tensor, gap: Optional[torch.Tensor]) -> torch.Tensor:
        if gap is None:
            return torch.zeros(*z.shape[:-1], self.gap_dim, device=z.device, dtype=z.dtype)
        return gap

    def forward_family(self, z: torch.Tensor, gap: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.family_head(torch.cat([z, self._gap(z, gap)], dim=-1))

    def _family_features(
        self,
        z: torch.Tensor,
        family_idx: Optional[torch.Tensor] = None,
        gap: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if family_idx is not None:
            return F.one_hot(family_idx, self.n_families).to(dtype=z.dtype, device=z.device)
        return F.softmax(self.forward_family(z, gap), dim=-1)

    def forward_reagent(
        self,
        z: torch.Tensor,
        *,
        family_idx: Optional[torch.Tensor] = None,
        gap: Optional[torch.Tensor] = None,
        atom_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        fam_feat = self._family_features(z, family_idx, gap)
        gap_t = self._gap(z, gap)
        if atom_emb is None:
            local = torch.zeros_like(z)
        else:
            local = atom_emb.max(dim=1).values
        return self.reagent_head(torch.cat([z, fam_feat, gap_t, local], dim=-1))

    def forward_sites(
        self,
        atom_emb: torch.Tensor,
        z: torch.Tensor,
        gap: Optional[torch.Tensor] = None,
        reagent_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-atom site logits. ``reagent_idx`` is the reagent being placed; ignored by reagent-blind heads."""
        n_atoms = atom_emb.size(1)
        z_exp = z.unsqueeze(1).expand(-1, n_atoms, -1)
        gap_exp = self._gap(z, gap).unsqueeze(1).expand(-1, n_atoms, -1)
        parts = [atom_emb, z_exp, gap_exp]
        if self.site_reagent_emb is not None:
            if reagent_idx is None:
                reag = torch.zeros(z.size(0), self.hidden, device=z.device, dtype=z.dtype)
            else:
                reag = self.site_reagent_emb(reagent_idx.to(device=z.device).view(-1))
            parts.append(reag.unsqueeze(1).expand(-1, n_atoms, -1))
        mixed = self.site_mix(torch.cat(parts, dim=-1))
        logits = self.site_head(mixed).squeeze(-1)
        if logits.dim() == 2 and logits.size(0) == 1:
            return logits.squeeze(0)
        return logits
