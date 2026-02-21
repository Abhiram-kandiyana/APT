import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except Exception:
    plt = None
    Line2D = None


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_percentile(values: np.ndarray, q: float, default: float = 0.0) -> float:
    values = np.asarray(values)
    if values.size == 0:
        return float(default)
    return float(np.percentile(values, q))


def _basin_size_entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.size == 0:
        return 0.0
    total = float(np.sum(counts))
    if total <= 0.0:
        return 0.0
    probs = counts / total
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _can_show_interactive_plots() -> bool:
    if plt is None:
        return False
    backend = str(plt.get_backend() or "").lower()
    # Non-interactive backends cannot open windows.
    return "agg" not in backend


def _label_from_caption(caption: str, label_map: Optional[List[str]]) -> Optional[int]:
    if caption is None:
        return None
    text = str(caption).strip()
    match = re.search(r"[cC]\s*:\s*([^\n\r]+)", text)
    token = ""
    if match:
        rhs = match.group(1).strip()
        rhs = rhs.lstrip("*'\"`_- ")
        for part in rhs.split():
            cleaned = part.strip(".,;:*\"'`_[]{}()")
            if cleaned:
                token = cleaned
                break
    if not token:
        return None
    if not label_map:
        return None
    labels_lower = [str(x).lower() for x in label_map]
    token_lower = token.lower()
    if token_lower in labels_lower:
        return labels_lower.index(token_lower)
    return None


class DiagnosticsAndTuner:
    """
    Per-round APT-DTS diagnostics + health checks + deterministic hyperparameter tuning.
    """

    def __init__(
        self,
        diagnostics_jsonl_path: str,
        diagnostic_outdir: str = "diagnostics",
        dataset_name: str = "dataset",
        diagnostic_mode: bool = False,
        show_interactive: bool = False,
        diagnostic_every: int = 2,
        diagnostic_seed: int = 0,
        max_images_per_panel: int = 40,
        clip_model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        clip_batch_size: int = 32,
        use_mutual_knn: bool = False,
    ):
        self.diagnostics_jsonl_path = diagnostics_jsonl_path
        self.diagnostic_outdir = diagnostic_outdir
        self.dataset_name = dataset_name
        self.diagnostic_mode = bool(diagnostic_mode)
        self.show_interactive = bool(show_interactive)
        self.diagnostic_every = max(1, int(diagnostic_every))
        self.diagnostic_seed = int(diagnostic_seed)
        self.max_images_per_panel = max(1, int(max_images_per_panel))
        self.clip_model_name = str(clip_model_name)
        self.clip_batch_size = int(clip_batch_size)
        self.use_mutual_knn = bool(use_mutual_knn)

    def _compute_health_trigger_thresholds(
        self,
        n_pool: int,
        batch_size: int,
        k_effective: int,
    ) -> Dict[str, float]:
        n_pool = max(1, int(n_pool))
        batch_size = max(1, int(batch_size))
        k_effective = max(1, int(k_effective))

        # Retune trigger sensitivity for smaller candidate pools (e.g., N_pool ~= 100).
        small_pool = n_pool <= 150
        if small_pool:
            c_overmerged_min = max(3, int(round(0.04 * n_pool)))
            top1_basin_frac_max = 0.70
            top5_basin_frac_max = 0.92

            c_fragmented_max = max(50, int(round(0.70 * n_pool)))
            basin_median_size_min = 6.0
            singleton_frac_max = 0.45
            tiny_frac_max = 0.60

            deg_mut_p10_min = max(4, int(math.ceil(0.35 * k_effective)))
            deg_mut_min_min = max(2, int(math.ceil(0.15 * k_effective)))

            diversity_min_unique_basins = max(3, int(math.ceil(0.25 * batch_size)))
        else:
            c_overmerged_min = 5
            top1_basin_frac_max = 0.60
            top5_basin_frac_max = 0.85

            c_fragmented_max = 80
            basin_median_size_min = 10.0
            singleton_frac_max = 0.30
            tiny_frac_max = 0.40

            deg_mut_p10_min = 10
            deg_mut_min_min = 5

            diversity_min_unique_basins = 4

        return {
            "c_overmerged_min": int(c_overmerged_min),
            "top1_basin_frac_max": float(top1_basin_frac_max),
            "top5_basin_frac_max": float(top5_basin_frac_max),
            "c_fragmented_max": int(c_fragmented_max),
            "basin_median_size_min": float(basin_median_size_min),
            "singleton_frac_max": float(singleton_frac_max),
            "tiny_frac_max": float(tiny_frac_max),
            "deg_mut_p10_min": int(deg_mut_p10_min),
            "deg_mut_min_min": int(deg_mut_min_min),
            "diversity_min_unique_basins": int(diversity_min_unique_basins),
        }

    def analyze_round(
        self,
        round_index: int,
        n_pool: int,
        batch_size: int,
        labeled_count: int,
        unlabeled_count: int,
        hyperparams: Dict[str, int],
        roots: np.ndarray,
        roots_eff: Optional[np.ndarray],
        basin_sizes: np.ndarray,
        basin_sizes_eff: Optional[np.ndarray],
        boundary_scores: np.ndarray,
        selected_indices: List[int],
        mutual_mask: np.ndarray,
        first_nn_distance: np.ndarray,
        state: Optional[Dict[str, Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        knn_indices: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None,
        parent: Optional[np.ndarray] = None,
        top1_nn_idx: Optional[np.ndarray] = None,
        selection_stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        selected_indices = [int(i) for i in selected_indices]
        roots = np.asarray(roots, dtype=np.int32)
        roots_eff = (
            np.asarray(roots_eff, dtype=np.int32)
            if roots_eff is not None
            else np.asarray(roots, dtype=np.int32)
        )
        basin_sizes = np.asarray(basin_sizes, dtype=np.int32)
        basin_sizes_eff = (
            np.asarray(basin_sizes_eff, dtype=np.int32)
            if basin_sizes_eff is not None
            else np.asarray(basin_sizes, dtype=np.int32)
        )
        boundary_scores = np.asarray(boundary_scores, dtype=np.float32)
        first_nn_distance = np.asarray(first_nn_distance, dtype=np.float32)
        mutual_mask = np.asarray(mutual_mask, dtype=bool)
        rho = np.asarray(rho, dtype=np.float32) if rho is not None else np.array([], dtype=np.float32)
        parent = np.asarray(parent, dtype=np.int32) if parent is not None else np.array([], dtype=np.int32)
        top1_nn_idx = np.asarray(top1_nn_idx, dtype=np.int32) if top1_nn_idx is not None else np.array([], dtype=np.int32)

        if len(boundary_scores) != int(n_pool):
            raise AssertionError(f"len(b_all)={len(boundary_scores)} does not match N_pool={int(n_pool)}")
        if len(roots) != int(n_pool):
            raise AssertionError(f"len(roots)={len(roots)} does not match N_pool={int(n_pool)}")
        if len(roots_eff) != int(n_pool):
            raise AssertionError(f"len(roots_eff)={len(roots_eff)} does not match N_pool={int(n_pool)}")
        if selected_indices:
            max_selected = max(selected_indices)
            if max_selected >= int(n_pool):
                raise AssertionError(f"selected index {max_selected} is out of range for N_pool={int(n_pool)}")

        deg_mut = np.sum(mutual_mask.astype(np.int32), axis=1) if mutual_mask.size else np.zeros(n_pool, dtype=np.int32)
        deg_mut_min = _as_int(np.min(deg_mut)) if deg_mut.size else 0
        deg_mut_p1 = _safe_percentile(deg_mut, 1, default=0.0)
        deg_mut_p5 = _safe_percentile(deg_mut, 5, default=0.0)
        deg_mut_p10 = _safe_percentile(deg_mut, 10, default=0.0)
        deg_mut_median = _safe_percentile(deg_mut, 50, default=0.0)
        deg_mut_p90 = _safe_percentile(deg_mut, 90, default=0.0)
        num_deg0 = int(np.sum(deg_mut == 0)) if deg_mut.size else 0
        frac_deg_lt_5 = float(np.mean(deg_mut < 5)) if deg_mut.size else 0.0

        unique_roots, root_counts = np.unique(roots, return_counts=True) if roots.size else (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
        c_basins = int(len(unique_roots))
        root_count_map = {int(root): int(count) for root, count in zip(unique_roots.tolist(), root_counts.tolist())}
        basin_size_all = np.asarray([root_count_map.get(int(r), 0) for r in roots.tolist()], dtype=np.int32)
        basin_sizes_sorted_desc = sorted([int(x) for x in root_counts.tolist()], reverse=True)
        basin_size_values = np.asarray(root_counts, dtype=np.int32)
        basin_min_size = int(min(basin_sizes_sorted_desc)) if basin_sizes_sorted_desc else 0
        basin_median_size = float(np.median(root_counts)) if root_counts.size else 0.0
        basin_max_size = int(max(basin_sizes_sorted_desc)) if basin_sizes_sorted_desc else 0
        singleton_frac = float(np.mean(root_counts == 1)) if root_counts.size else 0.0
        tiny_frac = float(np.mean(root_counts < int(hyperparams["mcluster_min"]))) if root_counts.size else 0.0
        points_tiny_frac = float(np.mean(basin_size_all < int(hyperparams["mcluster_min"]))) if basin_size_all.size else 0.0
        mass_top5_frac = float(np.sum(basin_sizes_sorted_desc[:5]) / max(n_pool, 1)) if basin_sizes_sorted_desc else 0.0
        top1_basin_id = int(unique_roots[np.argmax(root_counts)]) if root_counts.size else -1
        top5_basin_ids = [int(x) for x in unique_roots[np.argsort(root_counts)[::-1][:5]].tolist()] if root_counts.size else []
        top10_basins = [
            {"basin_id": int(root), "size": int(cnt)}
            for root, cnt in sorted(root_count_map.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ]
        num_singletons = int(np.sum(basin_size_values == 1)) if basin_size_values.size else 0
        num_tiny = int(np.sum(basin_size_values < int(hyperparams["mcluster_min"]))) if basin_size_values.size else 0
        size_p50 = _safe_percentile(basin_size_values, 50, default=0.0)
        size_p75 = _safe_percentile(basin_size_values, 75, default=0.0)
        size_p90 = _safe_percentile(basin_size_values, 90, default=0.0)
        basin_size_entropy = _basin_size_entropy_from_counts(root_counts)

        unique_roots_eff, root_counts_eff = np.unique(roots_eff, return_counts=True) if roots_eff.size else (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
        c_basins_eff = int(len(unique_roots_eff))
        root_count_map_eff = {
            int(root): int(count) for root, count in zip(unique_roots_eff.tolist(), root_counts_eff.tolist())
        }
        basin_size_all_eff = np.asarray([root_count_map_eff.get(int(r), 0) for r in roots_eff.tolist()], dtype=np.int32)
        basin_sizes_sorted_desc_eff = sorted([int(x) for x in root_counts_eff.tolist()], reverse=True)
        singleton_frac_eff = float(np.mean(root_counts_eff == 1)) if root_counts_eff.size else 0.0
        tiny_frac_eff = float(np.mean(root_counts_eff < int(hyperparams["mcluster_min"]))) if root_counts_eff.size else 0.0
        points_tiny_frac_eff = float(np.mean(basin_size_all_eff < int(hyperparams["mcluster_min"]))) if basin_size_all_eff.size else 0.0
        mass_top5_frac_eff = (
            float(np.sum(basin_sizes_sorted_desc_eff[:5]) / max(n_pool, 1)) if basin_sizes_sorted_desc_eff else 0.0
        )
        top1_basin_id_eff = int(unique_roots_eff[np.argmax(root_counts_eff)]) if root_counts_eff.size else -1
        top5_basin_ids_eff = (
            [int(x) for x in unique_roots_eff[np.argsort(root_counts_eff)[::-1][:5]].tolist()]
            if root_counts_eff.size
            else []
        )
        top10_basins_eff = [
            {"basin_id": int(root), "size": int(cnt)}
            for root, cnt in sorted(root_count_map_eff.items(), key=lambda kv: kv[1], reverse=True)[:10]
        ]

        b_mean = float(np.mean(boundary_scores)) if boundary_scores.size else 0.0
        b_std = float(np.std(boundary_scores)) if boundary_scores.size else 0.0
        b_p50 = _safe_percentile(boundary_scores, 50, default=0.0)
        b_p75 = _safe_percentile(boundary_scores, 75, default=0.0)
        b_p90 = _safe_percentile(boundary_scores, 90, default=0.0)
        b_p95 = _safe_percentile(boundary_scores, 95, default=0.0)
        b_p99 = _safe_percentile(boundary_scores, 99, default=0.0)
        b_p99_minus_p95 = float(b_p99 - b_p95)
        frac_b_ge_0_8 = float(np.mean(boundary_scores >= 0.8)) if boundary_scores.size else 0.0
        frac_b_ge_0_9 = float(np.mean(boundary_scores >= 0.9)) if boundary_scores.size else 0.0
        b_hist_counts = (
            np.histogram(np.clip(boundary_scores, 0.0, 1.0), bins=np.linspace(0.0, 1.0, 12))[0].astype(np.int32).tolist()
            if boundary_scores.size
            else [0] * 11
        )

        outlier_threshold = _safe_percentile(first_nn_distance, 99, default=0.0)
        outlier_flags = first_nn_distance > outlier_threshold
        d1_p50 = _safe_percentile(first_nn_distance, 50, default=0.0)
        d1_p90 = _safe_percentile(first_nn_distance, 90, default=0.0)
        d1_p95 = _safe_percentile(first_nn_distance, 95, default=0.0)
        d1_p99 = _safe_percentile(first_nn_distance, 99, default=0.0)
        frac_outliers = float(np.mean(first_nn_distance > outlier_threshold)) if first_nn_distance.size else 0.0

        valid_selected = [i for i in selected_indices if 0 <= i < boundary_scores.size]
        selected_boundary_scores = [float(boundary_scores[i]) for i in valid_selected]
        selected_roots_raw = [int(roots[i]) for i in valid_selected]
        selected_roots_eff = [int(roots_eff[i]) for i in valid_selected]
        selected_per_basin_counts_raw = {str(k): int(v) for k, v in Counter(selected_roots_raw).items()}
        selected_per_basin_counts_eff = {str(k): int(v) for k, v in Counter(selected_roots_eff).items()}
        selected_unique_basins_raw = int(len(set(selected_roots_raw)))
        selected_unique_basins_eff = int(len(set(selected_roots_eff)))
        selected_tiny_basin_count = int(
            sum(int(basin_size_all_eff[i]) < int(hyperparams["mcluster_min"]) for i in valid_selected)
        )
        selected_outlier_flag_count = int(sum(bool(outlier_flags[i]) for i in valid_selected))
        selected_boundary_std = (
            float(np.std(np.asarray(selected_boundary_scores, dtype=np.float32))) if selected_boundary_scores else 0.0
        )
        selected_basin_sizes_raw = (
            np.asarray([basin_size_all[i] for i in valid_selected], dtype=np.int32)
            if valid_selected
            else np.array([], dtype=np.int32)
        )
        selected_basin_sizes_eff = (
            np.asarray([basin_size_all_eff[i] for i in valid_selected], dtype=np.int32)
            if valid_selected
            else np.array([], dtype=np.int32)
        )
        selected_deg = np.asarray([deg_mut[i] for i in valid_selected], dtype=np.int32) if valid_selected else np.array([], dtype=np.int32)
        selected_d1 = np.asarray([first_nn_distance[i] for i in valid_selected], dtype=np.float32) if valid_selected else np.array([], dtype=np.float32)
        selected_in_largest_basin = int(np.sum(np.asarray(selected_roots_eff, dtype=np.int32) == int(top1_basin_id_eff))) if selected_roots_eff else 0
        top5_basin_set_eff = set(top5_basin_ids_eff)
        selected_in_top5_basins = int(sum(1 for x in selected_roots_eff if int(x) in top5_basin_set_eff))
        selected_in_tiny_basins = (
            int(np.sum(selected_basin_sizes_eff < int(hyperparams["mcluster_min"])))
            if selected_basin_sizes_eff.size
            else 0
        )
        selected_mean_basin_size = float(np.mean(selected_basin_sizes_eff)) if selected_basin_sizes_eff.size else 0.0
        selected_median_basin_size = float(np.median(selected_basin_sizes_eff)) if selected_basin_sizes_eff.size else 0.0
        selected_b_quantiles = {
            "p25": _safe_percentile(np.asarray(selected_boundary_scores, dtype=np.float32), 25, default=0.0),
            "p50": _safe_percentile(np.asarray(selected_boundary_scores, dtype=np.float32), 50, default=0.0),
            "p75": _safe_percentile(np.asarray(selected_boundary_scores, dtype=np.float32), 75, default=0.0),
        }
        selected_deg_mut_stats = {
            "mean": float(np.mean(selected_deg)) if selected_deg.size else 0.0,
            "median": float(np.median(selected_deg)) if selected_deg.size else 0.0,
        }
        selected_d1_stats = {
            "mean": float(np.mean(selected_d1)) if selected_d1.size else 0.0,
            "median": float(np.median(selected_d1)) if selected_d1.size else 0.0,
        }

        singleton_indicator = (basin_size_all == 1).astype(np.float32) if basin_size_all.size else np.array([], dtype=np.float32)
        if deg_mut.size and singleton_indicator.size and np.std(deg_mut.astype(np.float32)) > 0 and np.std(singleton_indicator) > 0:
            corr_deg_mut_singleton = float(np.corrcoef(deg_mut.astype(np.float32), singleton_indicator)[0, 1])
        else:
            corr_deg_mut_singleton = 0.0

        top1_basin_frac = float(basin_max_size / max(n_pool, 1))
        k_effective = int(mutual_mask.shape[1]) if mutual_mask.ndim == 2 else 1
        thresholds = self._compute_health_trigger_thresholds(
            n_pool=int(n_pool),
            batch_size=int(batch_size),
            k_effective=int(k_effective),
        )

        checks = {
            "overmerged": bool(
                (c_basins < int(thresholds["c_overmerged_min"]))
                or (top1_basin_frac > float(thresholds["top1_basin_frac_max"]))
                or (mass_top5_frac > float(thresholds["top5_basin_frac_max"]))
            ),
            "fragmented": bool(
                (c_basins > int(thresholds["c_fragmented_max"]))
                or (basin_median_size < float(thresholds["basin_median_size_min"]))
                or (singleton_frac > float(thresholds["singleton_frac_max"]))
                or (tiny_frac > float(thresholds["tiny_frac_max"]))
            ),
            "giant_and_dust": bool((top1_basin_frac >= 0.50) and (singleton_frac >= 0.40)),
            "mutual_sparse": bool(
                (deg_mut_p10 < float(thresholds["deg_mut_p10_min"]))
                or (deg_mut_min < int(thresholds["deg_mut_min_min"]))
            ),
            "outlier_heavy": bool((selected_outlier_flag_count >= 3) or (selected_tiny_basin_count >= 3)),
            "boundary_flat": bool((b_std < 0.05) or (b_p99_minus_p95 < 0.05) or (selected_boundary_std < 0.03)),
            "diversity_low": bool(selected_unique_basins_eff < int(thresholds["diversity_min_unique_basins"])),
        }

        updated_hparams, tuning_actions, updated_state, tuner_decision = self._tune_hyperparameters(
            round_index=int(round_index),
            current_hparams=hyperparams,
            checks=checks,
            state=state or {},
            current_metrics={
                "singleton_frac": float(singleton_frac),
                "basin_median_size": float(basin_median_size),
                "C": int(c_basins),
                "basin_size_entropy": float(basin_size_entropy),
                "selected_unique_basins_eff": int(selected_unique_basins_eff),
                "num_deg0": int(num_deg0),
            },
        )
        selected_cosine_summary = self._compute_selected_cosine_summary(
            embeddings=embeddings,
            selected_indices=valid_selected,
        )

        diagnostics: Dict[str, Any] = {
            "round_index": int(round_index),
            "N_pool": int(n_pool),
            "b": int(batch_size),
            "labeled_count": int(labeled_count),
            "unlabeled_count": int(unlabeled_count),
            "k": int(hyperparams["k"]),
            "k_rho": int(hyperparams["k_rho"]),
            "k_t": int(hyperparams["k_t"]),
            "k_b": int(hyperparams["k_b"]),
            "mcluster_min": int(hyperparams["mcluster_min"]),
            "c_tiny": int(hyperparams["c_tiny"]),
            "max_per_basin": int(hyperparams["max_per_basin"]),
            "deg_mut_min": int(deg_mut_min),
            "num_deg0": int(num_deg0),
            "deg_mut_p1": float(deg_mut_p1),
            "deg_mut_p5": float(deg_mut_p5),
            "deg_mut_p10": float(deg_mut_p10),
            "deg_mut_median": float(deg_mut_median),
            "deg_mut_p90": float(deg_mut_p90),
            "frac_deg_lt_5": float(frac_deg_lt_5),
            "corr_deg_mut_singleton": float(corr_deg_mut_singleton),
            "C": int(c_basins),
            "basin_min_size": int(basin_min_size),
            "basin_median_size": float(basin_median_size),
            "basin_max_size": int(basin_max_size),
            "singleton_frac": float(singleton_frac),
            "tiny_frac": float(tiny_frac),
            "points_tiny_frac": float(points_tiny_frac),
            "mass_top5_frac": float(mass_top5_frac),
            "top1_basin_frac": float(top1_basin_frac),
            "top5_basin_frac": float(mass_top5_frac),
            "top1_basin_id": int(top1_basin_id),
            "top5_basin_ids": [int(x) for x in top5_basin_ids],
            "basin_size_entropy": float(basin_size_entropy),
            "num_singletons": int(num_singletons),
            "num_tiny": int(num_tiny),
            "size_p50": float(size_p50),
            "size_p75": float(size_p75),
            "size_p90": float(size_p90),
            "top10_basins": top10_basins,
            "C_eff": int(c_basins_eff),
            "singleton_frac_eff": float(singleton_frac_eff),
            "tiny_frac_eff": float(tiny_frac_eff),
            "points_tiny_frac_eff": float(points_tiny_frac_eff),
            "mass_top5_frac_eff": float(mass_top5_frac_eff),
            "top1_basin_frac_eff": float((max(basin_sizes_sorted_desc_eff) if basin_sizes_sorted_desc_eff else 0) / max(n_pool, 1)),
            "top5_basin_frac_eff": float(mass_top5_frac_eff),
            "top1_basin_id_eff": int(top1_basin_id_eff),
            "top5_basin_ids_eff": [int(x) for x in top5_basin_ids_eff],
            "top10_basins_eff": top10_basins_eff,
            "b_mean": float(b_mean),
            "b_std": float(b_std),
            "frac_b_ge_0_8": float(frac_b_ge_0_8),
            "frac_b_ge_0_9": float(frac_b_ge_0_9),
            "b_p50": float(b_p50),
            "b_p75": float(b_p75),
            "b_p90": float(b_p90),
            "b_p95": float(b_p95),
            "b_p99": float(b_p99),
            "b_p99_minus_p95": float(b_p99_minus_p95),
            "b_hist": b_hist_counts,
            "d1_p50": float(d1_p50),
            "d1_p90": float(d1_p90),
            "d1_p95": float(d1_p95),
            "d1_p99": float(d1_p99),
            "frac_outliers": float(frac_outliers),
            "selected_indices": selected_indices,
            "selected_boundary_scores": selected_boundary_scores,
            "selected_unique_basins": int(selected_unique_basins_eff),
            "selected_unique_basins_raw": int(selected_unique_basins_raw),
            "selected_unique_basins_eff": int(selected_unique_basins_eff),
            "selected_per_basin_counts": selected_per_basin_counts_eff,
            "selected_per_basin_counts_raw": selected_per_basin_counts_raw,
            "selected_per_basin_counts_eff": selected_per_basin_counts_eff,
            "selected_tiny_basin_count": int(selected_tiny_basin_count),
            "selected_in_largest_basin": int(selected_in_largest_basin),
            "selected_in_top5_basins": int(selected_in_top5_basins),
            "selected_in_tiny_basins": int(selected_in_tiny_basins),
            "selected_mean_basin_size": float(selected_mean_basin_size),
            "selected_median_basin_size": float(selected_median_basin_size),
            "selected_b_quantiles": selected_b_quantiles,
            "selected_deg_mut_stats": selected_deg_mut_stats,
            "selected_d1_stats": selected_d1_stats,
            "selected_cos_sim_max": float(selected_cosine_summary["selected_cos_sim_max"]),
            "selected_cos_sim_median": float(selected_cosine_summary["selected_cos_sim_median"]),
            "selected_cos_sim_p90": float(selected_cosine_summary["selected_cos_sim_p90"]),
            "outlier_threshold": float(outlier_threshold),
            "selected_outlier_flag_count": int(selected_outlier_flag_count),
            "health_checks": checks,
            "triggers": checks,
            "trigger_thresholds": thresholds,
            "tuning_actions": tuning_actions,
            "tuner_decision": tuner_decision,
            "next_hyperparameters": updated_hparams,
            "tuner_state": updated_state,
            "debug_shapes": {
                "embeddings": list(np.asarray(embeddings).shape) if embeddings is not None else None,
                "knn_indices": list(np.asarray(knn_indices).shape) if knn_indices is not None else None,
                "rho": list(np.asarray(rho).shape) if rho is not None else None,
                "parent": list(np.asarray(parent).shape) if parent is not None else None,
                "top1_nn_idx": list(np.asarray(top1_nn_idx).shape) if top1_nn_idx is not None else None,
            },
        }
        if isinstance(selection_stats, dict):
            diagnostics.update(selection_stats)
        return diagnostics, updated_hparams, updated_state

    def run_round_diagnostics(
        self,
        round_index: int,
        n_pool: int,
        batch_size: int,
        labeled_count: int,
        unlabeled_count: int,
        hyperparams: Dict[str, int],
        roots: np.ndarray,
        roots_eff: Optional[np.ndarray],
        basin_sizes: np.ndarray,
        basin_sizes_eff: Optional[np.ndarray],
        boundary_scores: np.ndarray,
        selected_indices: List[int],
        selected_paths: List[str],
        candidate_paths: List[str],
        mutual_mask: np.ndarray,
        first_nn_distance: np.ndarray,
        state: Optional[Dict[str, Any]] = None,
        embeddings: Optional[np.ndarray] = None,
        knn_indices: Optional[np.ndarray] = None,
        knn_distances: Optional[np.ndarray] = None,
        rho: Optional[np.ndarray] = None,
        parent: Optional[np.ndarray] = None,
        top1_nn_idx: Optional[np.ndarray] = None,
        predicted_labels: Optional[List[Any]] = None,
        selected_true_labels: Optional[List[Any]] = None,
        label_map: Optional[List[str]] = None,
        labeled_paths_so_far: Optional[List[str]] = None,
        labeled_captions_so_far: Optional[List[str]] = None,
        unlabeled_paths_after_selection: Optional[List[str]] = None,
        selection_stats: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]:
        diagnostics, updated_hparams, updated_state = self.analyze_round(
            round_index=round_index,
            n_pool=n_pool,
            batch_size=batch_size,
            labeled_count=labeled_count,
            unlabeled_count=unlabeled_count,
            hyperparams=hyperparams,
            roots=roots,
            roots_eff=roots_eff,
            basin_sizes=basin_sizes,
            basin_sizes_eff=basin_sizes_eff,
            boundary_scores=boundary_scores,
            selected_indices=selected_indices,
            mutual_mask=mutual_mask,
            first_nn_distance=first_nn_distance,
            state=state,
            embeddings=embeddings,
            knn_indices=knn_indices,
            rho=rho,
            parent=parent,
            top1_nn_idx=top1_nn_idx,
            selection_stats=selection_stats,
        )

        normalized_selected_paths = self._normalize_selected_paths(
            selected_paths=selected_paths,
            selected_indices=selected_indices,
            candidate_paths=candidate_paths,
        )
        diagnostics["selected_ids"] = normalized_selected_paths

        if normalized_selected_paths and len(normalized_selected_paths) == len(selected_indices):
            selected_per_basin_eff = {}
            selected_per_basin_raw = {}
            for i, path in enumerate(normalized_selected_paths):
                idx = int(selected_indices[i])
                if 0 <= idx < len(roots):
                    selected_per_basin_raw[str(path)] = int(roots[idx])
                    eff_roots_arr = np.asarray(roots_eff) if roots_eff is not None else np.asarray(roots)
                    if idx < len(eff_roots_arr):
                        selected_per_basin_eff[str(path)] = int(eff_roots_arr[idx])
            diagnostics["selected_id_to_basin"] = selected_per_basin_eff
            diagnostics["selected_id_to_basin_raw"] = selected_per_basin_raw
            diagnostics["selected_id_to_basin_eff"] = selected_per_basin_eff

        round_dir = self._build_round_dir(round_index)
        diagnostics["round_dir"] = round_dir

        self._save_round_forensics_bundle(
            round_dir=round_dir,
            diagnostics=diagnostics,
            hyperparams=hyperparams,
            candidate_paths=candidate_paths,
            selected_indices=selected_indices,
            selected_paths=normalized_selected_paths,
            roots=np.asarray(roots),
            roots_eff=np.asarray(roots_eff) if roots_eff is not None else np.asarray(roots),
            boundary_scores=np.asarray(boundary_scores),
            rho=np.asarray(rho) if rho is not None else None,
            parent=np.asarray(parent) if parent is not None else None,
            mutual_mask=np.asarray(mutual_mask) if mutual_mask is not None else None,
            first_nn_distance=np.asarray(first_nn_distance),
            knn_indices=np.asarray(knn_indices) if knn_indices is not None else None,
            top1_nn_idx=np.asarray(top1_nn_idx) if top1_nn_idx is not None else None,
            embeddings=np.asarray(embeddings) if embeddings is not None else None,
        )

        checks = diagnostics.get("health_checks", {})
        any_trigger = any(bool(v) for v in checks.values())
        heavy_save = bool(self.diagnostic_mode and ((round_index % self.diagnostic_every == 0) or any_trigger))
        diagnostics["heavy_save_enabled"] = heavy_save

        if self.diagnostic_mode and round_dir is not None:
            outlier_threshold = float(diagnostics.get("outlier_threshold", 0.0))
            outlier_flags = np.asarray(first_nn_distance, dtype=np.float32) > outlier_threshold

            self._save_pca_basins(
                round_dir=round_dir,
                round_index=round_index,
                embeddings=embeddings,
                roots=np.asarray(roots),
                selected_indices=selected_indices,
                outlier_flags=outlier_flags,
                diagnostics=diagnostics,
            )
            if heavy_save:
                self._save_pca_boundary(
                    round_dir=round_dir,
                    embeddings=embeddings,
                    boundary_scores=np.asarray(boundary_scores),
                    selected_indices=selected_indices,
                )
                self._save_pca_predlabels(
                    round_dir=round_dir,
                    embeddings=embeddings,
                    predicted_labels=predicted_labels,
                    selected_indices=selected_indices,
                    label_map=label_map,
                )
                self._save_selected_panel(
                    round_dir=round_dir,
                    selected_indices=selected_indices,
                    selected_paths=normalized_selected_paths,
                    roots=np.asarray(roots),
                    boundary_scores=np.asarray(boundary_scores),
                    first_nn_distance=np.asarray(first_nn_distance),
                    outlier_flags=outlier_flags,
                    predicted_labels=predicted_labels,
                    selected_true_labels=selected_true_labels,
                    label_map=label_map,
                )
                self._save_basin_prototypes(
                    round_dir=round_dir,
                    candidate_paths=candidate_paths,
                    roots=np.asarray(roots),
                    rho=np.asarray(rho) if rho is not None else None,
                    boundary_scores=np.asarray(boundary_scores),
                    mcluster_min=int(hyperparams.get("mcluster_min", 20)),
                )
                self._save_boundary_bridges(
                    round_dir=round_dir,
                    selected_indices=selected_indices,
                    candidate_paths=candidate_paths,
                    roots=np.asarray(roots),
                    boundary_scores=np.asarray(boundary_scores),
                    knn_indices=np.asarray(knn_indices) if knn_indices is not None else None,
                    knn_distances=np.asarray(knn_distances) if knn_distances is not None else None,
                    mutual_mask=np.asarray(mutual_mask) if mutual_mask is not None else None,
                )

            # Purity diagnostics (Option A) are strictly post-selection and post-annotation only.
            # This block never feeds back into selection decisions (no leakage).
            purity = self._run_purity_diagnostics(
                round_dir=round_dir,
                round_index=round_index,
                hyperparams=hyperparams,
                label_map=label_map,
                labeled_paths_so_far=labeled_paths_so_far,
                labeled_captions_so_far=labeled_captions_so_far,
                unlabeled_paths_after_selection=unlabeled_paths_after_selection,
            )
            if purity:
                diagnostics["purity"] = purity

        self.append_jsonl(diagnostics)
        self.print_summary(diagnostics)

        return diagnostics, updated_hparams, updated_state

    def append_jsonl(self, diagnostics: Dict[str, Any]) -> None:
        out_dir = os.path.dirname(self.diagnostics_jsonl_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.diagnostics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(diagnostics, default=_json_default) + "\n")

    def _normalize_selected_paths(
        self,
        selected_paths: Optional[List[str]],
        selected_indices: List[int],
        candidate_paths: List[str],
    ) -> List[str]:
        out: List[str] = []
        selected_paths = [str(p) for p in (selected_paths or [])]
        if selected_paths and len(selected_paths) == len(selected_indices):
            return selected_paths
        for idx in selected_indices:
            i = int(idx)
            if 0 <= i < len(candidate_paths):
                out.append(str(candidate_paths[i]))
        return out

    def _compute_selected_cosine_summary(self, embeddings: Optional[np.ndarray], selected_indices: List[int]) -> Dict[str, float]:
        if embeddings is None:
            return {"selected_cos_sim_max": 0.0, "selected_cos_sim_median": 0.0, "selected_cos_sim_p90": 0.0}
        z = np.asarray(embeddings, dtype=np.float32)
        if z.ndim != 2 or z.shape[0] == 0:
            return {"selected_cos_sim_max": 0.0, "selected_cos_sim_median": 0.0, "selected_cos_sim_p90": 0.0}
        valid = [int(i) for i in selected_indices if 0 <= int(i) < z.shape[0]]
        if len(valid) < 2:
            return {"selected_cos_sim_max": 0.0, "selected_cos_sim_median": 0.0, "selected_cos_sim_p90": 0.0}
        zs = z[np.asarray(valid, dtype=np.int32)]
        norms = np.linalg.norm(zs, axis=1, keepdims=True)
        zs = zs / np.clip(norms, 1e-12, None)
        sim = zs @ zs.T
        iu = np.triu_indices(sim.shape[0], k=1)
        vals = np.asarray(sim[iu], dtype=np.float32)
        if vals.size == 0:
            return {"selected_cos_sim_max": 0.0, "selected_cos_sim_median": 0.0, "selected_cos_sim_p90": 0.0}
        return {
            "selected_cos_sim_max": float(np.max(vals)),
            "selected_cos_sim_median": float(np.median(vals)),
            "selected_cos_sim_p90": float(np.percentile(vals, 90)),
        }

    def _save_round_forensics_bundle(
        self,
        round_dir: str,
        diagnostics: Dict[str, Any],
        hyperparams: Dict[str, int],
        candidate_paths: List[str],
        selected_indices: List[int],
        selected_paths: List[str],
        roots: np.ndarray,
        roots_eff: np.ndarray,
        boundary_scores: np.ndarray,
        rho: Optional[np.ndarray],
        parent: Optional[np.ndarray],
        mutual_mask: Optional[np.ndarray],
        first_nn_distance: np.ndarray,
        knn_indices: Optional[np.ndarray],
        top1_nn_idx: Optional[np.ndarray],
        embeddings: Optional[np.ndarray],
    ) -> None:
        n_pool = int(diagnostics.get("N_pool", len(boundary_scores)))
        b_all = np.asarray(boundary_scores, dtype=np.float32)
        basin_id_all = np.asarray(roots, dtype=np.int32)
        basin_id_all_eff = np.asarray(roots_eff, dtype=np.int32)
        rho_all = np.asarray(rho, dtype=np.float32) if rho is not None else np.zeros(n_pool, dtype=np.float32)
        parent_all = np.asarray(parent, dtype=np.int32) if parent is not None else np.full(n_pool, -1, dtype=np.int32)
        is_root = (parent_all == -1).astype(np.uint8)
        if mutual_mask is not None and np.asarray(mutual_mask).size:
            deg_mut_all = np.sum(np.asarray(mutual_mask, dtype=np.int32), axis=1).astype(np.int32)
        else:
            deg_mut_all = np.zeros(n_pool, dtype=np.int32)
        d1_all = np.asarray(first_nn_distance, dtype=np.float32)
        unique_ids, counts = np.unique(basin_id_all, return_counts=True) if basin_id_all.size else (np.array([], dtype=np.int32), np.array([], dtype=np.int32))
        basin_count_map = {int(k): int(v) for k, v in zip(unique_ids.tolist(), counts.tolist())}
        basin_size_all = np.asarray([basin_count_map.get(int(r), 0) for r in basin_id_all.tolist()], dtype=np.int32)
        unique_ids_eff, counts_eff = np.unique(basin_id_all_eff, return_counts=True) if basin_id_all_eff.size else (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        )
        basin_count_map_eff = {int(k): int(v) for k, v in zip(unique_ids_eff.tolist(), counts_eff.tolist())}
        basin_size_all_eff = np.asarray([basin_count_map_eff.get(int(r), 0) for r in basin_id_all_eff.tolist()], dtype=np.int32)

        if top1_nn_idx is not None and np.asarray(top1_nn_idx).size == n_pool:
            top1_idx = np.asarray(top1_nn_idx, dtype=np.int32)
        elif knn_indices is not None and np.asarray(knn_indices).ndim == 2 and np.asarray(knn_indices).shape[1] > 0:
            top1_idx = np.asarray(knn_indices[:, 0], dtype=np.int32)
        else:
            top1_idx = np.full(n_pool, -1, dtype=np.int32)

        np.savez_compressed(
            os.path.join(round_dir, "pool_arrays.npz"),
            b_all=b_all,
            basin_id_all=basin_id_all,
            basin_id_all_eff=basin_id_all_eff,
            rho_all=rho_all,
            parent_all=parent_all,
            is_root=is_root,
            deg_mut_all=deg_mut_all,
            d1_all=d1_all,
            basin_size_all=basin_size_all,
            basin_size_all_eff=basin_size_all_eff,
            top1_nn_idx=top1_idx,
        )

        basin_size_by_id = {str(int(k)): int(v) for k, v in basin_count_map.items()}
        basin_size_by_id_eff = {str(int(k)): int(v) for k, v in basin_count_map_eff.items()}
        basin_map = {
            "basin_size_by_id": basin_size_by_id,
            "basin_size_by_id_eff": basin_size_by_id_eff,
            "top10_basins": diagnostics.get("top10_basins", []),
            "top10_basins_eff": diagnostics.get("top10_basins_eff", []),
            "basin_size_entropy": diagnostics.get("basin_size_entropy", 0.0),
            "C": diagnostics.get("C", 0),
            "C_eff": diagnostics.get("C_eff", 0),
        }
        with open(os.path.join(round_dir, "basin_sizes.json"), "w", encoding="utf-8") as f:
            json.dump(basin_map, f, indent=2, default=_json_default)

        valid_selected = [int(i) for i in selected_indices if 0 <= int(i) < n_pool]
        if len(selected_paths) == len(selected_indices):
            selected_paths_aligned = [
                str(p)
                for idx, p in zip(selected_indices, selected_paths)
                if 0 <= int(idx) < n_pool
            ]
        else:
            selected_paths_aligned = [
                str(candidate_paths[i]) for i in valid_selected if 0 <= int(i) < len(candidate_paths)
            ]
        selected_b_scores = [float(b_all[i]) for i in valid_selected]
        selected_basin_ids = [int(basin_id_all[i]) for i in valid_selected]
        selected_basin_ids_eff = [int(basin_id_all_eff[i]) for i in valid_selected]
        selected_cos = self._compute_selected_cosine_summary(embeddings=embeddings, selected_indices=valid_selected)
        selected_meta = {
            "selected_indices": valid_selected,
            "selected_boundary_scores": selected_b_scores,
            "selected_basin_ids": selected_basin_ids,
            "selected_basin_ids_eff": selected_basin_ids_eff,
            "selected_paths": selected_paths_aligned,
            "selected_pairwise_cosine_summary": selected_cos,
        }
        with open(os.path.join(round_dir, "selected_meta.json"), "w", encoding="utf-8") as f:
            json.dump(selected_meta, f, indent=2, default=_json_default)

    def print_summary(self, diagnostics: Dict[str, Any]) -> None:
        checks = diagnostics.get("health_checks", {})
        fired = [k for k, v in checks.items() if bool(v)]
        fired_text = ",".join(fired) if fired else "none"
        actions = diagnostics.get("tuning_actions", [])
        actions_text = ",".join(actions) if actions else "none"
        print(
            f"Round {diagnostics['round_index']} | "
            f"C={diagnostics['C']} | "
            f"top1={100.0 * float(diagnostics.get('top1_basin_frac', 0.0)):.1f}% top5={100.0 * float(diagnostics.get('top5_basin_frac', 0.0)):.1f}% | "
            f"singleton_frac={float(diagnostics.get('singleton_frac', 0.0)):.3f} | "
            f"deg0={int(diagnostics.get('num_deg0', 0))} p10={float(diagnostics.get('deg_mut_p10', 0.0)):.1f} | "
            f"frac_b>=0.8={float(diagnostics.get('frac_b_ge_0_8', 0.0)):.3f} frac_b>=0.9={float(diagnostics.get('frac_b_ge_0_9', 0.0)):.3f} | "
            f"sel_unique={diagnostics['selected_unique_basins']} sel_top5={int(diagnostics.get('selected_in_top5_basins', 0))} sel_tiny={int(diagnostics.get('selected_in_tiny_basins', 0))} | "
            f"sel_cos_max={float(diagnostics.get('selected_cos_sim_max', 0.0)):.4f} | "
            f"triggers={fired_text} "
            f"actions={actions_text}"
        )

    def _build_round_dir(self, round_index: int) -> str:
        os.makedirs(self.diagnostic_outdir, exist_ok=True)
        round_dir_name = f"round_{int(round_index):02d}"
        round_dir = os.path.join(self.diagnostic_outdir, round_dir_name)
        os.makedirs(round_dir, exist_ok=True)
        return round_dir

    def _compute_pca(self, embeddings: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if embeddings is None:
            return None
        z = np.asarray(embeddings, dtype=np.float32)
        if z.ndim != 2 or z.shape[0] == 0:
            return None
        if z.shape[0] == 1:
            return np.array([[0.0, 0.0]], dtype=np.float32)
        if z.shape[1] < 2:
            z = np.pad(z, ((0, 0), (0, 2 - z.shape[1])), mode="constant")
        pca = PCA(n_components=2, random_state=self.diagnostic_seed)
        return pca.fit_transform(z)

    def _finalize_figure(self, out_path: str):
        if plt is None:
            return
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        if self.diagnostic_mode and self.show_interactive and _can_show_interactive_plots():
            plt.show()
        plt.close()

    def _save_pca_basins(
        self,
        round_dir: str,
        round_index: int,
        embeddings: Optional[np.ndarray],
        roots: np.ndarray,
        selected_indices: List[int],
        outlier_flags: np.ndarray,
        diagnostics: Dict[str, Any],
    ) -> None:
        if plt is None:
            return

        pca_xy = self._compute_pca(embeddings)
        out_path = os.path.join(round_dir, "pca_basins.png")

        if pca_xy is None or pca_xy.shape[0] == 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.text(0.5, 0.5, "PCA unavailable", ha="center", va="center")
            ax.axis("off")
            self._finalize_figure(out_path)
            return

        fig, ax = plt.subplots(figsize=(9, 7))
        unique_roots, counts = np.unique(roots, return_counts=True) if roots.size else (np.array([], dtype=int), np.array([], dtype=int))
        top_root_order = [int(r) for r in unique_roots[np.argsort(counts)[::-1][:10]]]
        top_root_set = set(top_root_order)
        palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(top_root_order))))
        root_to_color = {rid: palette[i] for i, rid in enumerate(top_root_order)}

        others_mask = np.array([int(r) not in top_root_set for r in roots], dtype=bool)
        if np.any(others_mask):
            ax.scatter(pca_xy[others_mask, 0], pca_xy[others_mask, 1], s=14, c="#b7b7b7", alpha=0.35, label="other basins")

        for rid in top_root_order:
            mask = roots == rid
            ax.scatter(pca_xy[mask, 0], pca_xy[mask, 1], s=16, color=root_to_color[rid], alpha=0.7, label=f"basin {rid}")

        valid_selected = [i for i in selected_indices if 0 <= int(i) < pca_xy.shape[0]]
        if valid_selected:
            selected_colors = [root_to_color.get(int(roots[i]), "#7f7f7f") for i in valid_selected]
            ax.scatter(
                pca_xy[valid_selected, 0],
                pca_xy[valid_selected, 1],
                s=170,
                marker="*",
                c=selected_colors,
                edgecolors="black",
                linewidths=0.9,
                label="selected",
            )

        if outlier_flags.size == pca_xy.shape[0] and np.any(outlier_flags):
            outlier_indices = np.where(outlier_flags)[0]
            outlier_colors = [root_to_color.get(int(roots[i]), "#7f7f7f") for i in outlier_indices]
            ax.scatter(
                pca_xy[outlier_indices, 0],
                pca_xy[outlier_indices, 1],
                s=44,
                marker="^",
                c=outlier_colors,
                edgecolors="black",
                linewidths=0.7,
                label="outlier_flag",
            )

        text = (
            f"Round: {round_index}\n"
            f"Basins (C, root count): {diagnostics.get('C', 0)}\n"
            f"Singleton basin fraction: {diagnostics.get('singleton_frac', 0.0):.3f}\n"
            f"Mutual-kNN degree p10: {diagnostics.get('deg_mut_p10', 0.0):.1f}\n"
            f"Selected unique basins: {diagnostics.get('selected_unique_basins', 0)}"
        )
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )
        if Line2D is not None:
            marker_handles = [
                Line2D([0], [0], marker="*", color="w", markerfacecolor="#666666", markeredgecolor="black", markersize=10, linestyle="None", label="Selected (star, basin-colored)"),
                Line2D([0], [0], marker="^", color="w", markerfacecolor="#666666", markeredgecolor="black", markersize=8, linestyle="None", label="Outlier flag (triangle, basin-colored)"),
            ]
            ax.legend(handles=marker_handles, loc="lower right", fontsize=8, framealpha=0.9)

        ax.set_title("PCA: Basins + Selected")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        self._finalize_figure(out_path)

    def _save_pca_boundary(
        self,
        round_dir: str,
        embeddings: Optional[np.ndarray],
        boundary_scores: np.ndarray,
        selected_indices: List[int],
    ) -> None:
        if plt is None:
            return

        pca_xy = self._compute_pca(embeddings)
        out_path = os.path.join(round_dir, "pca_boundary.png")
        if pca_xy is None or pca_xy.shape[0] == 0:
            return

        fig, ax = plt.subplots(figsize=(9, 7))
        sc = ax.scatter(pca_xy[:, 0], pca_xy[:, 1], c=boundary_scores, s=18, cmap="viridis", alpha=0.8)
        plt.colorbar(sc, ax=ax, label="boundary score")

        valid_selected = [i for i in selected_indices if 0 <= int(i) < pca_xy.shape[0]]
        if valid_selected:
            selected_scores = np.asarray([boundary_scores[i] for i in valid_selected], dtype=np.float32)
            ax.scatter(
                pca_xy[valid_selected, 0],
                pca_xy[valid_selected, 1],
                s=170,
                marker="*",
                c=selected_scores,
                cmap="viridis",
                vmin=float(np.min(boundary_scores)) if boundary_scores.size else 0.0,
                vmax=float(np.max(boundary_scores)) if boundary_scores.size else 1.0,
                edgecolors="black",
                linewidths=0.9,
                label="selected (colored by boundary score)",
            )
            ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        ax.set_title("PCA: Boundary Scores")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        self._finalize_figure(out_path)

    def _save_pca_predlabels(
        self,
        round_dir: str,
        embeddings: Optional[np.ndarray],
        predicted_labels: Optional[List[Any]],
        selected_indices: List[int],
        label_map: Optional[List[str]] = None,
    ) -> None:
        if plt is None:
            return
        if not predicted_labels:
            return

        pca_xy = self._compute_pca(embeddings)
        out_path = os.path.join(round_dir, "pca_predlabels.png")
        if pca_xy is None or pca_xy.shape[0] == 0:
            return

        preds = np.asarray(predicted_labels, dtype=object)
        if preds.shape[0] != pca_xy.shape[0]:
            return

        valid_mask = np.array([p is not None and str(p) != "" for p in preds], dtype=bool)
        fig, ax = plt.subplots(figsize=(9, 7))

        if np.any(~valid_mask):
            ax.scatter(pca_xy[~valid_mask, 0], pca_xy[~valid_mask, 1], s=14, c="#c9c9c9", alpha=0.35, label="unknown/no pred")

        labels_lower = [str(x).lower() for x in (label_map or [])]

        def _norm_pred(p: Any) -> str:
            if p is None:
                return ""
            if isinstance(p, (int, np.integer)):
                idx = int(p)
                if 0 <= idx < len(labels_lower):
                    return labels_lower[idx]
                return str(idx)
            s = str(p).strip().lower()
            if s in labels_lower:
                return s
            if s.isdigit():
                idx = int(s)
                if 0 <= idx < len(labels_lower):
                    return labels_lower[idx]
            return s

        normalized_preds = np.array([_norm_pred(p) for p in preds], dtype=object)
        unique_labels = sorted({str(p) for p in normalized_preds[valid_mask] if str(p) != ""})

        # Strongly separated colors for binary case; fallback palette for >2 classes.
        if len(unique_labels) == 2:
            label_to_color = {}
            for lbl in unique_labels:
                l = str(lbl).lower()
                if "wild" in l:
                    label_to_color[lbl] = "#1f77b4"  # blue
                elif "lurcher" in l:
                    label_to_color[lbl] = "#d62728"  # red
            remaining = [lbl for lbl in unique_labels if lbl not in label_to_color]
            for i, lbl in enumerate(remaining):
                label_to_color[lbl] = "#1f77b4" if i == 0 else "#d62728"
        else:
            palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_labels))))
            label_to_color = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

        for lbl in unique_labels:
            mask = np.array([str(p) == lbl for p in normalized_preds], dtype=bool)
            pretty = lbl
            if labels_lower and lbl in labels_lower:
                pretty = label_map[labels_lower.index(lbl)]
            ax.scatter(pca_xy[mask, 0], pca_xy[mask, 1], s=22, color=label_to_color[lbl], alpha=0.9, label=f"pred: {pretty}")

        valid_selected = [i for i in selected_indices if 0 <= int(i) < pca_xy.shape[0]]
        if valid_selected:
            ax.scatter(
                pca_xy[valid_selected, 0],
                pca_xy[valid_selected, 1],
                s=170,
                marker="*",
                c="black",
                edgecolors="white",
                linewidths=0.7,
                label="selected",
            )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

        ax.set_title("PCA: Predicted Labels")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        self._finalize_figure(out_path)

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        try:
            with Image.open(str(image_path)) as img:
                return img.convert("RGB")
        except Exception:
            return None

    def _save_image_grid(self, items: List[Dict[str, str]], out_path: str, title: str, ncols: int = 5) -> None:
        if plt is None:
            return
        if not items:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, "No items to display", ha="center", va="center")
            ax.axis("off")
            ax.set_title(title)
            self._finalize_figure(out_path)
            return

        max_items = min(len(items), self.max_images_per_panel)
        items = items[:max_items]
        ncols = max(1, int(ncols))
        nrows = int(math.ceil(max_items / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.4))
        axes = np.asarray(axes).reshape(-1)

        for i, ax in enumerate(axes):
            ax.axis("off")
            if i >= max_items:
                continue
            item = items[i]
            img = self._load_image(item.get("image_path", ""))
            if img is not None:
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, "image\nmissing", ha="center", va="center", fontsize=9)
            ax.set_xlabel(item.get("caption", ""), fontsize=7)

        fig.suptitle(title)
        self._finalize_figure(out_path)

    def _save_selected_panel(
        self,
        round_dir: str,
        selected_indices: List[int],
        selected_paths: List[str],
        roots: np.ndarray,
        boundary_scores: np.ndarray,
        first_nn_distance: np.ndarray,
        outlier_flags: np.ndarray,
        predicted_labels: Optional[List[Any]],
        selected_true_labels: Optional[List[Any]],
        label_map: Optional[List[str]],
    ) -> None:
        items = []
        for pos, idx in enumerate(selected_indices):
            idx_i = int(idx)
            if not (0 <= idx_i < roots.size):
                continue

            pred = None
            if predicted_labels is not None and idx_i < len(predicted_labels):
                pred = predicted_labels[idx_i]

            true_lbl = None
            if selected_true_labels is not None and pos < len(selected_true_labels):
                true_lbl = selected_true_labels[pos]
                if label_map and isinstance(true_lbl, (int, np.integer)) and 0 <= int(true_lbl) < len(label_map):
                    true_lbl = label_map[int(true_lbl)]

            d1 = float(first_nn_distance[idx_i]) if idx_i < first_nn_distance.size else 0.0
            out = bool(outlier_flags[idx_i]) if idx_i < outlier_flags.size else False

            path = str(selected_paths[pos]) if pos < len(selected_paths) else ""
            caption = (
                f"idx={idx_i} basin={int(roots[idx_i])}\n"
                f"b={float(boundary_scores[idx_i]):.3f} d1={d1:.3f} out={int(out)}\n"
                f"pred={pred} true={true_lbl}"
            )
            items.append({"image_path": path, "caption": caption})

        self._save_image_grid(
            items=items,
            out_path=os.path.join(round_dir, "selected_panel.png"),
            title="Selected Samples",
            ncols=5,
        )

    def _save_basin_prototypes(
        self,
        round_dir: str,
        candidate_paths: List[str],
        roots: np.ndarray,
        rho: Optional[np.ndarray],
        boundary_scores: np.ndarray,
        mcluster_min: int,
    ) -> None:
        basin_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i, root in enumerate(roots.tolist()):
            basin_to_indices[int(root)].append(int(i))

        non_tiny = [(root, idxs) for root, idxs in basin_to_indices.items() if len(idxs) >= int(mcluster_min)]
        non_tiny.sort(key=lambda x: len(x[1]), reverse=True)

        items: List[Dict[str, str]] = []
        for basin_id, idxs in non_tiny[:5]:
            idxs_arr = np.asarray(idxs, dtype=np.int32)
            rho_vals = np.asarray(rho[idxs_arr], dtype=np.float32) if rho is not None and rho.size > 0 else np.zeros(len(idxs), dtype=np.float32)
            b_vals = np.asarray(boundary_scores[idxs_arr], dtype=np.float32)

            if idxs_arr.size == 0:
                continue

            central_idx = int(idxs_arr[int(np.argmax(rho_vals))])
            order = np.argsort(rho_vals)
            median_idx = int(idxs_arr[int(order[len(order) // 2])])
            high_b_idx = int(idxs_arr[int(np.argmax(b_vals))])

            picks = [
                ("central", central_idx),
                ("median_rho", median_idx),
                ("high_boundary", high_b_idx),
            ]
            for tag, pick_idx in picks:
                if not (0 <= pick_idx < len(candidate_paths)):
                    continue
                items.append(
                    {
                        "image_path": str(candidate_paths[pick_idx]),
                        "caption": (
                            f"basin={basin_id} size={len(idxs)} {tag}\n"
                            f"rho={float(rho[pick_idx]) if rho is not None and pick_idx < len(rho) else 0.0:.3f} "
                            f"b={float(boundary_scores[pick_idx]):.3f}"
                        ),
                    }
                )

        self._save_image_grid(
            items=items,
            out_path=os.path.join(round_dir, "basin_prototypes.png"),
            title="Basin Prototypes (Top Non-Tiny Basins)",
            ncols=3,
        )

    def _save_boundary_bridges(
        self,
        round_dir: str,
        selected_indices: List[int],
        candidate_paths: List[str],
        roots: np.ndarray,
        boundary_scores: np.ndarray,
        knn_indices: Optional[np.ndarray],
        knn_distances: Optional[np.ndarray],
        mutual_mask: Optional[np.ndarray],
    ) -> None:
        if knn_indices is None or knn_indices.size == 0:
            return

        items: List[Dict[str, str]] = []
        for idx in selected_indices:
            i = int(idx)
            if not (0 <= i < len(candidate_paths)):
                continue

            items.append(
                {
                    "image_path": str(candidate_paths[i]),
                    "caption": f"selected idx={i} basin={int(roots[i])} b={float(boundary_scores[i]):.3f}",
                }
            )

            nbrs = np.asarray(knn_indices[i], dtype=np.int32)
            dists = np.asarray(knn_distances[i], dtype=np.float32) if knn_distances is not None and knn_distances.size else np.zeros(len(nbrs), dtype=np.float32)

            if self.use_mutual_knn and mutual_mask is not None and mutual_mask.size:
                valid = np.asarray(mutual_mask[i], dtype=bool)
                nbrs = nbrs[valid]
                dists = dists[valid]

            added = 0
            for j, d in zip(nbrs.tolist(), dists.tolist()):
                if not (0 <= int(j) < len(candidate_paths)):
                    continue
                if int(roots[j]) == int(roots[i]):
                    continue
                items.append(
                    {
                        "image_path": str(candidate_paths[int(j)]),
                        "caption": f"nbr idx={int(j)} basin={int(roots[j])} d={float(d):.3f}",
                    }
                )
                added += 1
                if added >= 3:
                    break

        self._save_image_grid(
            items=items,
            out_path=os.path.join(round_dir, "boundary_bridges.png"),
            title="Boundary Bridges: Selected + Cross-Basin Neighbors",
            ncols=4,
        )

    def _run_purity_diagnostics(
        self,
        round_dir: str,
        round_index: int,
        hyperparams: Dict[str, int],
        label_map: Optional[List[str]],
        labeled_paths_so_far: Optional[List[str]],
        labeled_captions_so_far: Optional[List[str]],
        unlabeled_paths_after_selection: Optional[List[str]],
    ) -> Optional[Dict[str, Any]]:
        if not self.diagnostic_mode:
            return None

        labeled_paths = [str(p) for p in (labeled_paths_so_far or [])]
        captions = [str(c) for c in (labeled_captions_so_far or [])]
        unlabeled_paths = [str(p) for p in (unlabeled_paths_after_selection or [])]

        if not labeled_paths or not captions or len(labeled_paths) != len(captions):
            return None

        labeled_labels: List[Optional[int]] = [_label_from_caption(c, label_map) for c in captions]
        known_labeled = [(p, y) for p, y in zip(labeled_paths, labeled_labels) if y is not None and int(y) >= 0]
        if not known_labeled:
            return None

        labeled_paths = [p for p, _ in known_labeled]
        labeled_labels = [int(y) for _, y in known_labeled]

        diag_paths = unlabeled_paths + labeled_paths
        if len(diag_paths) < 2:
            return None

        try:
            from dts_sampling import score_candidates_with_dts

            _, meta_diag = score_candidates_with_dts(
                image_paths=diag_paths,
                k=int(hyperparams["k"]),
                k_rho=int(hyperparams["k_rho"]),
                k_t=int(hyperparams["k_t"]),
                k_b=int(hyperparams["k_b"]),
                use_mutual_knn=self.use_mutual_knn,
                mcluster_min=int(hyperparams["mcluster_min"]),
                clip_model_name=self.clip_model_name,
                clip_batch_size=self.clip_batch_size,
                embedding_cache_path=os.path.join(round_dir, "purity_embeddings.npz"),
            )
        except Exception as e:
            return {"error": f"purity_failed: {e}"}

        roots_diag = np.asarray(meta_diag.get("root_eff", meta_diag.get("root", [])), dtype=np.int32)
        if roots_diag.size != len(diag_paths):
            return {"error": "purity_failed: root size mismatch"}

        offset = len(unlabeled_paths)
        min_labeled_per_basin = 3 if int(round_index) <= 2 else 5
        basin_label_counts: Dict[int, Counter] = defaultdict(Counter)

        for i, y in enumerate(labeled_labels):
            basin_id = int(roots_diag[offset + i])
            basin_label_counts[basin_id][int(y)] += 1

        basin_rows = []
        for basin_id, cnt in basin_label_counts.items():
            total = int(sum(cnt.values()))
            if total < min_labeled_per_basin:
                continue
            max_count = int(max(cnt.values())) if cnt else 0
            purity = float(max_count / max(total, 1))
            probs = np.array([v / total for v in cnt.values()], dtype=np.float64)
            entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
            counts_named = {}
            for lbl_idx, v in cnt.items():
                key = str(label_map[lbl_idx]) if label_map and 0 <= lbl_idx < len(label_map) else str(lbl_idx)
                counts_named[key] = int(v)
            basin_rows.append(
                {
                    "basin_id": int(basin_id),
                    "labeled_count": int(total),
                    "purity": float(purity),
                    "entropy": float(entropy),
                    "label_counts": counts_named,
                }
            )

        basin_rows.sort(key=lambda x: (x["purity"], -x["labeled_count"]))
        num_basins = len(basin_rows)
        mean_purity = float(np.mean([x["purity"] for x in basin_rows])) if basin_rows else 0.0
        frac_ge_08 = float(np.mean([x["purity"] >= 0.8 for x in basin_rows])) if basin_rows else 0.0

        report = {
            "round_index": int(round_index),
            "min_labeled_per_basin": int(min_labeled_per_basin),
            "num_purity_basins": int(num_basins),
            "mean_purity": float(mean_purity),
            "frac_purity_ge_0.8": float(frac_ge_08),
            "lowest_purity_basins": basin_rows[:5],
        }

        json_path = os.path.join(round_dir, "purity_report.json")
        txt_path = os.path.join(round_dir, "purity_report.txt")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_json_default)

        lines = [
            f"round: {report['round_index']}",
            f"min_labeled_per_basin: {report['min_labeled_per_basin']}",
            f"num_purity_basins: {report['num_purity_basins']}",
            f"mean_purity: {report['mean_purity']:.4f}",
            f"frac_purity_ge_0.8: {report['frac_purity_ge_0.8']:.4f}",
            "lowest_purity_basins:",
        ]
        for row in report["lowest_purity_basins"]:
            lines.append(
                f"  basin={row['basin_id']} labeled={row['labeled_count']} purity={row['purity']:.4f} "
                f"entropy={row['entropy']:.4f} counts={row['label_counts']}"
            )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return report

    def _tune_hyperparameters(
        self,
        round_index: int,
        current_hparams: Dict[str, int],
        checks: Dict[str, bool],
        state: Dict[str, Any],
        current_metrics: Dict[str, Any],
    ) -> Tuple[Dict[str, int], List[str], Dict[str, Any], Dict[str, Any]]:
        updated = {
            "k": int(current_hparams["k"]),
            "k_rho": int(current_hparams["k_rho"]),
            "k_t": int(current_hparams["k_t"]),
            "k_b": int(current_hparams["k_b"]),
            "mcluster_min": int(current_hparams["mcluster_min"]),
            "c_tiny": int(current_hparams["c_tiny"]),
            "max_per_basin": int(current_hparams["max_per_basin"]),
        }
        actions: List[str] = []
        freeze_tuning_rounds = int(state.get("freeze_tuning_rounds", 0))
        chosen_trigger = "none"
        applied_action = "none"
        reverted = False
        revert_reason = ""

        prev_hparams = state.get("prev_hparams")
        prev_C = state.get("prev_C")
        prev_action_applied = bool(state.get("prev_action_applied", False))

        C_now = int(current_metrics.get("C", 0))
        entropy_now = float(current_metrics.get("basin_size_entropy", 0.0))

        collapse_gt_50 = bool(prev_C is not None and float(prev_C) > 0 and C_now < 0.5 * float(prev_C))
        guard_fail = bool(C_now < 3 or entropy_now < 0.3 or collapse_gt_50)

        if prev_action_applied and guard_fail and isinstance(prev_hparams, dict):
            updated = {
                "k": int(prev_hparams.get("k", current_hparams["k"])),
                "k_rho": int(prev_hparams.get("k_rho", current_hparams["k_rho"])),
                "k_t": int(prev_hparams.get("k_t", current_hparams["k_t"])),
                "k_b": int(prev_hparams.get("k_b", current_hparams["k_b"])),
                "mcluster_min": int(prev_hparams.get("mcluster_min", current_hparams["mcluster_min"])),
                "c_tiny": int(prev_hparams.get("c_tiny", current_hparams["c_tiny"])),
                "max_per_basin": int(prev_hparams.get("max_per_basin", current_hparams["max_per_basin"])),
            }
            freeze_tuning_rounds = 2
            reverted = True
            chosen_trigger = "revert_guard"
            reasons = []
            if C_now < 3:
                reasons.append(f"C<{3}")
            if entropy_now < 0.3:
                reasons.append("basin_size_entropy<0.3")
            if collapse_gt_50:
                reasons.append("unique_basins_collapse>50%")
            revert_reason = ",".join(reasons) if reasons else "guard"
            applied_action = "revert_to_prev_hparams"
            actions.append(f"GUARD_REVERT:{revert_reason}")
        elif int(round_index) <= 2:
            chosen_trigger = "warmup_freeze"
            applied_action = "none"
            actions.append(f"warmup_freeze:round={int(round_index)}")
        elif freeze_tuning_rounds > 0:
            actions.append(f"freeze_tuning:remaining={freeze_tuning_rounds}")
            freeze_tuning_rounds = max(0, freeze_tuning_rounds - 1)
            chosen_trigger = "freeze_tuning"
            applied_action = "none"
        else:
            giant_and_dust = bool(checks.get("giant_and_dust", False))
            fragmented_only = bool(checks.get("fragmented", False) and not checks.get("overmerged", False) and not giant_and_dust)
            overmerged_only = bool(checks.get("overmerged", False) and not checks.get("fragmented", False) and not giant_and_dust)

            priority_order = [
                ("outlier_heavy", bool(checks.get("outlier_heavy", False))),
                ("boundary_flat", bool(checks.get("boundary_flat", False))),
                ("giant_and_dust", bool(giant_and_dust)),
                ("fragmented_only", bool(fragmented_only)),
                ("overmerged_only", bool(overmerged_only)),
                ("diversity_low", bool(checks.get("diversity_low", False))),
            ]

            chosen = next((name for name, fired in priority_order if fired), None)
            if chosen is not None:
                chosen_trigger = str(chosen)
                if chosen == "giant_and_dust":
                    prev = int(updated["k_rho"])
                    updated["k_rho"] = min(80, prev + 10)
                    if updated["k_rho"] != prev:
                        applied_action = f"k_rho:{prev}->{updated['k_rho']}"
                        actions.append(f"giant_and_dust:k_rho {prev}->{updated['k_rho']}")
                elif chosen == "outlier_heavy":
                    prev = int(updated["mcluster_min"])
                    updated["mcluster_min"] = min(50, max(10, prev + 10))
                    if updated["mcluster_min"] != prev:
                        applied_action = f"mcluster_min:{prev}->{updated['mcluster_min']}"
                        actions.append(f"outlier_heavy:mcluster_min {prev}->{updated['mcluster_min']}")
                elif chosen == "boundary_flat":
                    prev = int(updated["k_b"])
                    updated["k_b"] = min(50, prev + 5)
                    if updated["k_b"] != prev:
                        applied_action = f"k_b:{prev}->{updated['k_b']}"
                        actions.append(f"boundary_flat:k_b {prev}->{updated['k_b']}")
                elif chosen == "fragmented_only":
                    if bool(checks.get("mutual_sparse", False)):
                        prev = int(updated["k"])
                        updated["k"] = min(200, prev + 20)
                        if updated["k"] != prev:
                            applied_action = f"k:{prev}->{updated['k']}"
                            actions.append(f"fragmented_only+mutual_sparse:k {prev}->{updated['k']}")
                    else:
                        prev = int(updated["k_t"])
                        updated["k_t"] = min(80, prev + 10)
                        if updated["k_t"] != prev:
                            applied_action = f"k_t:{prev}->{updated['k_t']}"
                            actions.append(f"fragmented_only:k_t {prev}->{updated['k_t']}")
                elif chosen == "overmerged_only":
                    prev = int(updated["k_t"])
                    updated["k_t"] = max(20, prev - 10)
                    if updated["k_t"] != prev:
                        applied_action = f"k_t:{prev}->{updated['k_t']}"
                        actions.append(f"overmerged_only:k_t {prev}->{updated['k_t']}")
                elif chosen == "diversity_low":
                    prev = int(updated["max_per_basin"])
                    updated["max_per_basin"] = 1
                    if updated["max_per_basin"] != prev:
                        applied_action = f"max_per_basin:{prev}->{updated['max_per_basin']}"
                        actions.append(f"diversity_low:max_per_basin {prev}->{updated['max_per_basin']}")

        updated["k"] = int(np.clip(updated["k"], 50, 200))
        updated["k_rho"] = int(np.clip(updated["k_rho"], 20, 80))
        updated["k_t"] = int(np.clip(updated["k_t"], 20, 80))
        updated["k_b"] = int(np.clip(updated["k_b"], 20, 50))
        updated["mcluster_min"] = int(np.clip(updated["mcluster_min"], 10, 50))
        updated["c_tiny"] = 1
        updated["max_per_basin"] = 1 if updated["max_per_basin"] == 1 else 2
        if updated["k_t"] > updated["k"]:
            prev_kt = updated["k_t"]
            updated["k_t"] = updated["k"]
            if updated["k_t"] != prev_kt:
                actions.append(f"consistency:k_t {prev_kt}->{updated['k_t']}")

        updated_state = {
            "overmerged_streak": int(state.get("overmerged_streak", 0) + (1 if checks.get("overmerged", False) else 0)),
            "fragmented_streak": int(state.get("fragmented_streak", 0) + (1 if checks.get("fragmented", False) else 0)),
            "boundary_flat_streak": int(state.get("boundary_flat_streak", 0) + (1 if checks.get("boundary_flat", False) else 0)),
            "cooldown_k": int(state.get("cooldown_k", 0)),
            "freeze_tuning_rounds": int(freeze_tuning_rounds),
            "prev_singleton_frac": float(current_metrics.get("singleton_frac", 0.0)),
            "prev_basin_median_size": float(current_metrics.get("basin_median_size", 0.0)),
            "prev_C": int(current_metrics.get("C", 0)),
            "prev_basin_size_entropy": float(current_metrics.get("basin_size_entropy", 0.0)),
            "prev_selected_unique_basins_eff": int(current_metrics.get("selected_unique_basins_eff", 0)),
            "prev_action_applied": bool(applied_action != "none"),
            "prev_num_deg0": int(current_metrics.get("num_deg0", 0)),
            "prev_hparams": {
                "k": int(current_hparams["k"]),
                "k_rho": int(current_hparams["k_rho"]),
                "k_t": int(current_hparams["k_t"]),
                "k_b": int(current_hparams["k_b"]),
                "mcluster_min": int(current_hparams["mcluster_min"]),
                "c_tiny": int(current_hparams["c_tiny"]),
                "max_per_basin": int(current_hparams["max_per_basin"]),
            },
        }
        decision = {
            "chosen_trigger": str(chosen_trigger),
            "applied_action": str(applied_action),
            "reverted": bool(reverted),
            "revert_reason": str(revert_reason),
        }
        return updated, actions, updated_state, decision
