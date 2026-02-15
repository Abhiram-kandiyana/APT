import os
import tempfile
import unittest

import numpy as np

from dts_diagnostics import DiagnosticsAndTuner


class TestDTSDiagnosticsAndTuning(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.diagnostics_path = os.path.join(self.tmpdir.name, "aptdts_diagnostics.jsonl")
        self.tuner = DiagnosticsAndTuner(self.diagnostics_path)
        self.base_hparams = {
            "k": 80,
            "k_rho": 30,
            "k_t": 30,
            "k_b": 20,
            "mcluster_min": 20,
            "c_tiny": 1,
            "max_per_basin": 2,
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def _common_inputs(self, n: int):
        mutual_mask = np.ones((n, 10), dtype=bool)
        first_nn_distance = np.linspace(0.01, 0.2, n, dtype=np.float32)
        boundary_scores = np.linspace(0.1, 0.9, n, dtype=np.float32)
        selected_indices = list(range(min(10, n)))
        return mutual_mask, first_nn_distance, boundary_scores, selected_indices

    def test_warmup_freeze_no_hparam_change(self):
        n = 100
        roots = np.zeros(n, dtype=np.int32)
        roots_eff = roots.copy()
        basin_sizes = np.full(n, n, dtype=np.int32)
        basin_sizes_eff = basin_sizes.copy()
        mutual_mask, first_nn_distance, _, _ = self._common_inputs(n)
        # Avoid boundary_flat so giant_and_dust is selected by priority.
        boundary_scores = np.concatenate(
            [
                np.linspace(0.0, 0.2, 90, dtype=np.float32),
                np.linspace(0.8, 1.0, 10, dtype=np.float32),
            ]
        )
        selected_indices = [0, 20, 40, 60, 80, 90, 91, 92, 93, 94]

        diagnostics, next_hparams, next_state = self.tuner.analyze_round(
            round_index=1,
            n_pool=n,
            batch_size=10,
            labeled_count=20,
            unlabeled_count=80,
            hyperparams=self.base_hparams,
            roots=roots,
            roots_eff=roots_eff,
            basin_sizes=basin_sizes,
            basin_sizes_eff=basin_sizes_eff,
            boundary_scores=boundary_scores,
            selected_indices=selected_indices,
            mutual_mask=mutual_mask,
            first_nn_distance=first_nn_distance,
            state={},
        )
        self.assertEqual(next_hparams, self.base_hparams)
        self.assertEqual(diagnostics["tuner_decision"]["chosen_trigger"], "warmup_freeze")
        self.assertEqual(diagnostics["tuner_decision"]["applied_action"], "none")
        self.assertFalse(diagnostics["tuner_decision"]["reverted"])
        self.assertEqual(int(next_state["freeze_tuning_rounds"]), 0)

    def test_giant_and_dust_updates_only_k_rho_after_warmup(self):
        checks = {
            "overmerged": True,
            "fragmented": True,
            "giant_and_dust": True,
            "mutual_sparse": True,
            "outlier_heavy": False,
            "boundary_flat": False,
            "diversity_low": False,
        }
        next_hparams, _, _, decision = self.tuner._tune_hyperparameters(
            round_index=3,
            current_hparams=self.base_hparams,
            checks=checks,
            state={},
            current_metrics={
                "singleton_frac": 0.5,
                "basin_median_size": 1.0,
                "C": 7,
                "basin_size_entropy": 0.8,
                "selected_unique_basins_eff": 4,
                "num_deg0": 0,
            },
        )

        self.assertEqual(decision["chosen_trigger"], "giant_and_dust")
        self.assertTrue(str(decision["applied_action"]).startswith("k_rho:"))

        # Only k_rho changes under giant_and_dust.
        self.assertEqual(next_hparams["k_rho"], self.base_hparams["k_rho"] + 10)
        self.assertEqual(next_hparams["k"], self.base_hparams["k"])
        self.assertEqual(next_hparams["k_t"], self.base_hparams["k_t"])
        self.assertEqual(next_hparams["k_b"], self.base_hparams["k_b"])
        self.assertEqual(next_hparams["mcluster_min"], self.base_hparams["mcluster_min"])

    def test_revert_guard_restores_previous_hparams(self):
        n = 20
        roots = np.zeros(n, dtype=np.int32)  # C=1 triggers guard.
        roots_eff = roots.copy()
        basin_sizes = np.full(n, n, dtype=np.int32)
        basin_sizes_eff = basin_sizes.copy()
        mutual_mask, first_nn_distance, boundary_scores, selected_indices = self._common_inputs(n)

        prev_hparams = {
            "k": 60,
            "k_rho": 20,
            "k_t": 20,
            "k_b": 20,
            "mcluster_min": 10,
            "c_tiny": 1,
            "max_per_basin": 2,
        }
        state = {
            "prev_hparams": prev_hparams,
            "prev_C": 10,
            "prev_basin_size_entropy": 1.0,
            "prev_action_applied": True,
            "freeze_tuning_rounds": 0,
        }

        diagnostics, next_hparams, next_state = self.tuner.analyze_round(
            round_index=3,
            n_pool=n,
            batch_size=10,
            labeled_count=20,
            unlabeled_count=80,
            hyperparams=self.base_hparams,
            roots=roots,
            roots_eff=roots_eff,
            basin_sizes=basin_sizes,
            basin_sizes_eff=basin_sizes_eff,
            boundary_scores=boundary_scores,
            selected_indices=selected_indices,
            mutual_mask=mutual_mask,
            first_nn_distance=first_nn_distance,
            state=state,
        )
        self.assertEqual(next_hparams, prev_hparams)
        self.assertTrue(diagnostics["tuner_decision"]["reverted"])
        self.assertEqual(diagnostics["tuner_decision"]["chosen_trigger"], "revert_guard")
        self.assertEqual(int(next_state["freeze_tuning_rounds"]), 2)

    def test_effective_basin_metrics_reduce_c_without_collapsing(self):
        # Raw: 8 basins (one big + seven tiny); Eff: 4 basins (not collapsed to 1-2).
        roots = np.array(
            [10, 10, 10, 10, 10, 11, 12, 13, 14, 15, 16, 17],
            dtype=np.int32,
        )
        roots_eff = np.array(
            [10, 10, 10, 10, 10, 10, 10, 13, 13, 15, 15, 17],
            dtype=np.int32,
        )
        n = int(len(roots))
        basin_sizes = np.ones(n, dtype=np.int32)
        basin_sizes_eff = np.ones(n, dtype=np.int32)
        mutual_mask, first_nn_distance, boundary_scores, selected_indices = self._common_inputs(n)

        diagnostics, _, _ = self.tuner.analyze_round(
            round_index=3,
            n_pool=n,
            batch_size=6,
            labeled_count=20,
            unlabeled_count=80,
            hyperparams=self.base_hparams,
            roots=roots,
            roots_eff=roots_eff,
            basin_sizes=basin_sizes,
            basin_sizes_eff=basin_sizes_eff,
            boundary_scores=boundary_scores,
            selected_indices=selected_indices[:6],
            mutual_mask=mutual_mask,
            first_nn_distance=first_nn_distance,
            state={},
        )

        self.assertGreater(diagnostics["C"], diagnostics["C_eff"])
        self.assertGreater(diagnostics["C_eff"], 2)
        self.assertIn("selected_per_basin_counts_raw", diagnostics)
        self.assertIn("selected_per_basin_counts_eff", diagnostics)
        # `selected_per_basin_counts` must use effective ids.
        self.assertEqual(diagnostics["selected_per_basin_counts"], diagnostics["selected_per_basin_counts_eff"])


if __name__ == "__main__":
    unittest.main()
