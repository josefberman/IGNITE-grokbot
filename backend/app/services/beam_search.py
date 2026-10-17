"""Beam search controller following the high-level architecture diagram."""

from __future__ import annotations

import logging
import math
import random
from typing import Callable, List, Optional

from backend.app.domain.models import (
    BeamParameters,
    BeamSearchTree,
    BeamTreeNode,
    CandidateMolecule,
    JobStatus,
    ModificationMode,
    NoveltyStatus,
    SearchProgress,
    TargetProperties,
)
from backend.app.services.modification import ModificationMechanism, PlannerUnavailableError
from backend.app.services.novelty import NoveltyService
from backend.app.services.run_logger import build_run_log
from backend.app.services.property_estimator import PropertyEstimator
from backend.app.services.safety import estimate_safety
from backend.app.services.scoring import score_candidate, rank_candidates
from backend.app.services.seed_selector import SeedLibrary

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[SearchProgress, Optional[CandidateMolecule]], None]

# Hard-reject only truly critical hazards; energetic azides are scored, not gated.
CRITICAL_MOTIFS = {"peroxide", "invalid", "long_n_chain"}


class BeamSearchController:
    def __init__(
        self,
        seeds: SeedLibrary,
        estimator: PropertyEstimator,
        novelty: NoveltyService,
        modification: Optional[ModificationMechanism] = None,
    ):
        self.seeds = seeds
        self.estimator = estimator
        self.novelty = novelty
        self.modification = modification or ModificationMechanism()
        self._stop = False
        self.last_seed_smiles: List[str] = []
        self.last_candidates: List[CandidateMolecule] = []
        self.last_mode: Optional[ModificationMode] = None
        self.last_run_log: Optional[dict] = None

    def request_stop(self) -> None:
        self._stop = True

    @staticmethod
    def _select_survivors(
        pool: List[CandidateMolecule],
        top_k: int,
        *,
        explore: bool,
    ) -> List[CandidateMolecule]:
        """Keep top_k by score, reserving slots for generated children while exploring."""
        ranked = rank_candidates(pool)
        if not explore or top_k <= 1:
            return ranked[:top_k]

        children = [c for c in ranked if c.generation > 0]
        seeds = [c for c in ranked if c.generation == 0]
        reserve = min(len(children), max(1, math.ceil(top_k / 2)))
        chosen: List[CandidateMolecule] = []
        seen_ids = set()

        for c in children:
            if len(chosen) >= reserve:
                break
            chosen.append(c)
            seen_ids.add(c.id)

        for c in ranked:
            if len(chosen) >= top_k:
                break
            if c.id in seen_ids:
                continue
            chosen.append(c)
            seen_ids.add(c.id)

        # Prefer filling with remaining seeds if somehow short
        for c in seeds:
            if len(chosen) >= top_k:
                break
            if c.id in seen_ids:
                continue
            chosen.append(c)

        return rank_candidates(chosen)[:top_k]

    def run(
        self,
        job_id: str,
        targets: TargetProperties,
        beam: BeamParameters,
        on_progress: Optional[ProgressCallback] = None,
        mode: ModificationMode = ModificationMode.RULE_BASED,
    ) -> tuple[CandidateMolecule, List[CandidateMolecule], SearchProgress, BeamSearchTree]:
        self._stop = False
        self.last_mode = mode
        self.last_seed_smiles = []
        self.last_candidates = []
        self.last_run_log = None
        tree = BeamSearchTree()
        rng = random.Random(beam.random_seed)
        self.modification.rng = rng
        if mode == ModificationMode.PLANNER and not self.modification.learned.available:
            raise PlannerUnavailableError(
                self.modification.learned.unavailable_reason
                or "Learned planner is unavailable: torch or the planner checkpoint is missing."
            )

        # Honor GUI toggles
        self.estimator.literature.enabled = bool(beam.enable_literature)
        self.novelty.pubchem_enabled = bool(beam.enable_literature and self.novelty.pubchem_enabled)
        self.novelty.openalex_enabled = bool(beam.enable_literature and self.novelty.openalex_enabled)
        if not beam.enable_patent_search:
            self.novelty.epo_ops_key = ""
            self.novelty.lens_api_key = ""

        target_dict = targets.as_dict()
        seed_pool = self.seeds.select_seeds(
            targets, mape_threshold=max(beam.mape_threshold, 25.0), max_seeds=1
        )
        if not seed_pool:
            raise ValueError("No seed molecules available")

        for s in seed_pool:
            s.safety = estimate_safety(s.canonical_smiles)
            if beam.novelty_as_gate:
                s.novelty = self.novelty.evaluate(s.canonical_smiles)
                if s.novelty.status == NoveltyStatus.KNOWN:
                    s.rejected = True
                    s.rejection_reason = "novelty_gate"
            s.score = score_candidate(s, target_dict, explore=True)
            s.mape = s.score.mape
        seed_pool = [s for s in seed_pool if not s.rejected]
        if not seed_pool:
            raise ValueError("No seed molecules available after novelty gate")

        beam_nodes = sorted(seed_pool, key=lambda c: c.score.total if c.score else 1e9)[:1]
        best = beam_nodes[0]
        self.last_seed_smiles = [c.canonical_smiles for c in beam_nodes]
        seen = {c.canonical_smiles for c in beam_nodes}
        catalog = {c.canonical_smiles: c for c in beam_nodes}
        tried_reagents: dict[str, set[str]] = {c.canonical_smiles: set() for c in beam_nodes}
        exhausted: set[str] = set()
        rejected = 0
        generated = 0
        source_counts = {"dataset": 0, "literature": 0, "xgboost": 0, "other": 0}

        def _tally(cand: CandidateMolecule) -> None:
            for est in cand.properties.values():
                key = est.source.value if hasattr(est.source, "value") else str(est.source)
                if key in source_counts:
                    source_counts[key] += 1
                else:
                    source_counts["other"] += 1
                break  # one count per molecule

        for s in beam_nodes:
            _tally(s)

        def _tree_node(c: CandidateMolecule, *, in_beam: bool) -> BeamTreeNode:
            mod = c.modification
            return BeamTreeNode(
                id=c.id,
                parent_id=c.parent_id,
                generation=c.generation,
                canonical_smiles=c.canonical_smiles,
                mape=c.mape,
                score_total=c.score.total if c.score else None,
                reaction_family=mod.reaction_family if mod else None,
                planner=mod.planner if mod else None,
                in_beam=in_beam,
            )

        def _register(c: CandidateMolecule, *, in_beam: bool) -> None:
            tree.nodes.append(_tree_node(c, in_beam=in_beam))

        def _sync_beam_layer(nodes: List[CandidateMolecule]) -> None:
            for n in tree.nodes:
                n.in_beam = False
            ids = []
            for c in nodes:
                ids.append(c.id)
                for i, existing in enumerate(tree.nodes):
                    if existing.id == c.id:
                        tree.nodes[i] = _tree_node(c, in_beam=True)
                        break
                else:
                    _register(c, in_beam=True)
            tree.beam_layers.append(ids)

        _sync_beam_layer(beam_nodes)

        def emit(iteration: int, status: JobStatus, message: str) -> SearchProgress:
            prog = SearchProgress(
                job_id=job_id,
                status=status,
                iteration=iteration,
                max_iterations=beam.depth,
                beam_size=len(beam_nodes),
                candidates_generated=generated,
                rejected_count=rejected,
                best_mape=best.mape,
                best_score=best.score.total if best.score else None,
                message=message,
            )
            if on_progress:
                on_progress(prog, best)
            return prog

        progress = emit(
            0,
            JobStatus.RUNNING,
            f"Seeded with {len(beam_nodes)} molecules (mode={mode.value}, lit={beam.enable_literature})",
        )

        for iteration in range(1, beam.depth + 1):
            if self._stop:
                progress = emit(iteration - 1, JobStatus.CANCELLED, "Cancelled")
                break

            exploring = best.mape > beam.mape_threshold

            # Early stop only when threshold met AND we have finished at least one expansion
            # if seeds already meet threshold — allow returning seed; otherwise always expand once.
            if (
                beam.early_stop_on_mape
                and best.mape <= beam.mape_threshold
                and iteration > 1
            ):
                max_gen = max((n.generation for n in tree.nodes), default=0)
                progress = emit(
                    iteration - 1,
                    JobStatus.COMPLETED,
                    f"MAPE threshold reached ({max_gen} tree generations from seed)",
                )
                break
            if best.mape <= beam.mape_threshold and iteration == 1 and best.generation == 0:
                # Still run one modification pass to seek novelty/improvement
                exploring = True

            children: List[CandidateMolecule] = []
            # Beam parents first, then other molecules already in the tree, deepest
            # first. Each parent skips reactions it has already used.
            expand_from: List[CandidateMolecule] = []
            expand_ids: set[str] = set()

            def _queue(parent: CandidateMolecule) -> None:
                if parent.id in expand_ids or parent.canonical_smiles in exhausted:
                    return
                if parent.generation >= beam.depth:
                    return
                expand_ids.add(parent.id)
                expand_from.append(parent)

            for parent in beam_nodes:
                _queue(parent)
            parent_budget = max(beam.top_k, 1) * 2
            for parent in sorted(catalog.values(), key=lambda c: -c.generation):
                if len(expand_from) >= parent_budget:
                    break
                _queue(parent)

            for parent in expand_from:
                tried = tried_reagents.setdefault(parent.canonical_smiles, set())
                tried_before = len(tried)
                expansions = self.modification.expand(
                    parent.canonical_smiles,
                    parent.property_values(),
                    target_dict,
                    n=beam.width,
                    forbidden=seen,
                    skip_reagents=tried,
                    mode=mode,
                )
                if not expansions and len(tried) == tried_before:
                    exhausted.add(parent.canonical_smiles)
                for child_smi, mod in expansions:
                    generated += 1
                    if child_smi in seen:
                        rejected += 1
                        continue
                    props = self.estimator.estimate(
                        child_smi, prefer_literature=beam.enable_literature
                    )
                    if props is None:
                        rejected += 1
                        continue
                    cand = CandidateMolecule(
                        smiles=child_smi,
                        canonical_smiles=child_smi,
                        generation=parent.generation + 1,
                        parent_id=parent.id,
                        parent_smiles=parent.canonical_smiles,
                        properties=props,
                        modification=mod,
                        lineage=list(parent.lineage) + [parent.canonical_smiles],
                    )
                    cand.safety = estimate_safety(child_smi)
                    critical = set(cand.safety.unstable_motifs) & CRITICAL_MOTIFS
                    if critical or (not cand.safety.is_acceptable and "peroxide" in cand.safety.unstable_motifs):
                        cand.rejected = True
                        cand.rejection_reason = f"safety_gate:{','.join(sorted(critical)) or 'peroxide'}"
                        rejected += 1
                        continue
                    if cand.safety.score >= 0.95:
                        cand.rejected = True
                        cand.rejection_reason = "safety_score_extreme"
                        rejected += 1
                        continue

                    if beam.novelty_as_gate:
                        cand.novelty = self.novelty.evaluate(child_smi)
                        if cand.novelty.status == NoveltyStatus.KNOWN:
                            cand.rejected = True
                            cand.rejection_reason = "novelty_gate"
                            rejected += 1
                            continue

                    cand.score = score_candidate(cand, target_dict, explore=exploring)
                    cand.mape = cand.score.mape
                    seen.add(child_smi)
                    children.append(cand)
                    catalog[child_smi] = cand
                    tried_reagents.setdefault(child_smi, set())
                    _register(cand, in_beam=False)
                    _tally(cand)

            pool = beam_nodes + children
            for c in pool:
                # Re-score with current explore mode for fair comparison
                c.score = score_candidate(c, target_dict, explore=exploring)
                c.mape = c.score.mape

            beam_nodes = self._select_survivors(pool, beam.top_k, explore=exploring)
            _sync_beam_layer(beam_nodes)
            if beam_nodes:
                # Final best: prefer lowest total among survivors; break seed ties by MAPE
                # Children vs seed: edit-free score; children among themselves: full total.
                candidate_best = rank_candidates(beam_nodes)[0]
                if (
                    best.score is None
                    or rank_candidates([best, candidate_best])[0] is candidate_best
                    and candidate_best is not best
                    or (exploring and best.generation == 0 and candidate_best.generation > 0
                        and candidate_best.mape <= best.mape + 1.0)
                ):
                    best = candidate_best

            src_msg = (
                f"xgboost={source_counts['xgboost']} lit={source_counts['literature']} "
                f"dataset={source_counts['dataset']}"
            )
            progress = emit(
                iteration,
                JobStatus.RUNNING,
                f"Iteration {iteration}: +{len(children)} children, {src_msg}",
            )

        else:
            max_gen = max((n.generation for n in tree.nodes), default=0)
            progress = emit(
                beam.depth,
                JobStatus.COMPLETED,
                f"Depth limit reached ({max_gen} tree generations from seed)",
            )

        if progress.status == JobStatus.RUNNING:
            progress.status = JobStatus.COMPLETED

        tree.max_generation = max((n.generation for n in tree.nodes), default=0)
        tree.iterations_completed = progress.iteration

        # Final ranking without explore bias
        for c in beam_nodes:
            c.score = score_candidate(c, target_dict, explore=False)
            c.mape = c.score.mape
        top = rank_candidates(beam_nodes)
        best = top[0] if top else best
        best.score = score_candidate(best, target_dict, explore=False)
        best.mape = best.score.mape
        self.last_candidates = list(catalog.values())
        self.last_run_log = build_run_log(
            job_id=job_id,
            mode=mode,
            targets=targets,
            seed_smiles=self.last_seed_smiles,
            candidates=self.last_candidates,
            status=progress.status.value,
        )
        for c in top:
            c.strip_internal_signals()
        best.strip_internal_signals()
        tree.best_id = best.id
        return best, top, progress, tree
