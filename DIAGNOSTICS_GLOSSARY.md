# APT-DTS Diagnostics Glossary

This document defines the diagnostics terms and plots produced by APT-DTS.

## Core Structure Terms
- `Basin`: A cluster induced by density-tree parent pointers. Each point follows its parent pointer to a root; all points with the same root belong to one basin.
- `Root`: A point with no valid uphill parent under DTS rules.
- `C`: Number of basins (equivalently, number of unique roots).
- `N_pool`: Number of candidate/unlabeled points used for current-round DTS geometry.

## DTS Hyperparameters (Logged Each Round)
- `k`: Graph kNN size used to build neighborhood graph.
- `k_rho`: Number of neighbors used for local density `rho_i`.
- `k_t`: Rank used for threshold radius `t_i` (distance to `k_t`-th neighbor).
- `k_b`: Number of neighbors used by boundary score calculation.
- `mcluster_min`: Minimum basin size threshold; smaller basins are considered tiny.
- `c_tiny`: Maximum selected samples allowed from one tiny basin.
- `max_per_basin`: Per-basin diversity cap used in selection.

## Degree / Basin / Boundary Metrics
- `deg_mut_min`: Minimum mutual-kNN degree in pool.
- `deg_mut_p10`: 10th percentile of mutual-kNN degree.
- `deg_mut_median`: Median mutual-kNN degree.
- `deg_mut_p90`: 90th percentile of mutual-kNN degree.
- `basin_min_size`: Smallest basin size.
- `basin_median_size`: Median basin size.
- `basin_max_size`: Largest basin size.
- `singleton_frac`: Fraction of basins with size 1.
- `tiny_frac`: Fraction of basins with size `< mcluster_min`.
- `mass_top5_frac`: Fraction of total pool mass contained in the 5 largest basins.
- `b_mean`: Mean boundary score over pool.
- `b_std`: Standard deviation of boundary scores.
- `b_p95`: 95th percentile boundary score.
- `b_p99`: 99th percentile boundary score.
- `b_p99_minus_p95`: Tail spread of boundary scores; low values indicate a flat tail.

## Selection Metrics
- `selected_ids`: File paths of selected samples.
- `selected_boundary_scores`: Boundary scores for selected samples.
- `selected_unique_basins`: Number of distinct basins represented by selected set.
- `selected_tiny_basin_count`: Count of selected samples from tiny basins.
- `selected_per_basin_counts`: Selected count per basin.

## Outlier Diagnostics
- `d1(i)`: First-neighbor distance in standard kNN.
- `outlier_threshold`: 99th percentile of `d1` in the pool.
- `outlier_flag(i)`: `True` if `d1(i) > outlier_threshold`.
- `selected_outlier_flag_count`: Number of outlier-flagged samples in selected set.

## Trigger Definitions
A trigger fires when its condition is true:
- `overmerged`: `C < 5` OR `basin_max_size/N > 0.60` OR `mass_top5_frac > 0.85`
- `fragmented`: `C > 80` OR `basin_median_size < 10` OR `singleton_frac > 0.30` OR `tiny_frac > 0.40`
- `mutual_sparse`: `deg_mut_p10 < 10` OR `deg_mut_min < 5`
- `outlier_heavy`: `selected_outlier_flag_count >= 3` OR `selected_tiny_basin_count >= 3`
- `boundary_flat`: `b_std < 0.05` OR `b_p99_minus_p95 < 0.05` OR `std(selected_boundary_scores) < 0.03`
- `diversity_low`: `selected_unique_basins <= 3`

## Plot Guide

### `pca_basins.png`
- 2D PCA of candidate embeddings.
- Point colors: top basins by size get unique colors; others are gray.
- Selected samples: star markers, basin-colored.
- Outlier flags: triangle markers, basin-colored.
- Text box includes short descriptive metric names.

### `pca_boundary.png`
- 2D PCA with point color from boundary score colormap.
- Selected samples are stars colored by their own boundary score value.

### `pca_predlabels.png`
- 2D PCA with points colored by predicted label where available.
- Unknown/no-pred points are gray.
- Legend shows label names and colors (binary case uses blue/red for readability).

### `selected_panel.png`
- Grid of selected images with captions:
  - sample index, basin id,
  - boundary score,
  - first-neighbor distance,
  - outlier flag,
  - predicted label (if available),
  - true label (after annotation).

### `basin_prototypes.png`
For each of the largest non-tiny basins, shows three representative samples:
- `central`: highest `rho` point (density-central under DTS proxy).
- `median_rho`: median-by-`rho` point in that basin.
- `high_boundary`: highest boundary-score point in that basin.

Note: this is **not** geometric centroid nearest-neighbor selection; it is DTS-metric based.

### `boundary_bridges.png`
For each selected sample, shows:
- the selected image, and
- up to 3 nearest neighbors from **different basins**.

Purpose: visually validate that high-boundary selections connect/lie near cross-basin interfaces (bridges), rather than being isolated noise.

## Purity Diagnostics (Post-Selection Only, No Leakage)
- Executed only after current round selection and after labels are known.
- Uses `Z_diag = unlabeled_after_selection + all_labeled_so_far` and reruns DTS clustering for diagnostics only.
- Purity stats computed only on basins with sufficient labeled count:
  - rounds 1-2: minimum 3 labeled samples per basin,
  - later rounds: minimum 5.
- Reported metrics:
  - `num_purity_basins`,
  - `mean_purity`,
  - `frac_purity_ge_0.8`,
  - lowest-purity basin table with label counts.

These purity values are logged/reported only and are never used to alter same-round selection.
