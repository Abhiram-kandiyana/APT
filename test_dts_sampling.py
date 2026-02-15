import unittest
from unittest.mock import patch

import numpy as np
import os
import tempfile

from dts_sampling import _compute_effective_basin_ids, score_candidates_with_dts


class _DummyEmbedder:
    def __init__(self, embeddings: np.ndarray):
        self._embeddings = embeddings
        self.calls = 0

    def embed_image_paths(self, image_paths, batch_size=32):
        self.calls += 1
        return self._embeddings[:len(image_paths)]


class TestDTSSampling(unittest.TestCase):
    def test_single_candidate_returns_zero_boundary(self):
        scores, meta = score_candidates_with_dts(["img_0.jpg"])
        self.assertEqual(scores.shape, (1,))
        self.assertAlmostEqual(float(scores[0]), 0.0, places=6)
        self.assertEqual(int(meta["cluster_size"][0]), 1)

    @patch("dts_sampling._get_clip_embedder")
    def test_scores_shape_and_range(self, mock_get_embedder):
        # Two compact clusters with one bridge point.
        embeddings = np.array([
            [1.0, 0.0],
            [0.98, 0.02],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.02, 0.98],
            [0.05, 0.95],
            [0.7, 0.7],
        ], dtype=np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        mock_get_embedder.return_value = _DummyEmbedder(embeddings)
        image_paths = [f"img_{i}.jpg" for i in range(len(embeddings))]

        scores, meta = score_candidates_with_dts(
            image_paths=image_paths,
            k=5,
            k_rho=3,
            k_t=3,
            k_b=3,
            clip_batch_size=4,
        )

        self.assertEqual(scores.shape, (len(image_paths),))
        self.assertEqual(meta["rho"].shape, (len(image_paths),))
        self.assertEqual(meta["root"].shape, (len(image_paths),))
        self.assertEqual(meta["cluster_size"].shape, (len(image_paths),))

        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))
        self.assertTrue(np.all(meta["cluster_size"] >= 1))

    @patch("dts_sampling._get_clip_embedder")
    def test_mutual_knn_mode(self, mock_get_embedder):
        # Construct points where mutual edges are a strict subset of directed kNN.
        embeddings = np.array([
            [1.0, 0.0],
            [0.99, 0.01],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ], dtype=np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        mock_get_embedder.return_value = _DummyEmbedder(embeddings)
        image_paths = [f"img_{i}.jpg" for i in range(len(embeddings))]

        scores, meta = score_candidates_with_dts(
            image_paths=image_paths,
            k=3,
            k_rho=2,
            k_t=2,
            k_b=2,
            use_mutual_knn=True,
        )

        self.assertEqual(scores.shape, (len(image_paths),))
        self.assertEqual(meta["root"].shape, (len(image_paths),))
        self.assertTrue(np.all(scores >= 0.0))
        self.assertTrue(np.all(scores <= 1.0))

    @patch("dts_sampling._get_clip_embedder")
    def test_embedding_cache_reused(self, mock_get_embedder):
        embeddings = np.array([
            [1.0, 0.0],
            [0.95, 0.05],
            [0.0, 1.0],
            [0.05, 0.95],
        ], dtype=np.float32)
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        dummy = _DummyEmbedder(embeddings)
        mock_get_embedder.return_value = dummy
        image_paths = [f"img_{i}.jpg" for i in range(len(embeddings))]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "round_clip_embeddings.npz")

            score_candidates_with_dts(
                image_paths=image_paths,
                k=3,
                k_rho=2,
                k_t=2,
                k_b=2,
                embedding_cache_path=cache_path,
            )
            self.assertEqual(dummy.calls, 1)
            self.assertTrue(os.path.exists(cache_path))

            score_candidates_with_dts(
                image_paths=image_paths,
                k=3,
                k_rho=2,
                k_t=2,
                k_b=2,
                embedding_cache_path=cache_path,
            )
            self.assertEqual(dummy.calls, 1)

    def test_effective_basin_squash_reduces_c_without_collapsing(self):
        # Raw: three non-tiny basins + three tiny singleton basins.
        roots = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 10, 11, 12], dtype=np.int32)
        parents = np.array([-1] * len(roots), dtype=np.int32)
        # Nearest-neighbor fallback maps tiny basins onto different large basins.
        top1_nn_idx = np.array([1, 0, 1, 4, 3, 4, 7, 6, 7, 0, 3, 6], dtype=np.int32)

        roots_eff, size_eff = _compute_effective_basin_ids(
            roots=roots,
            parents=parents,
            top1_nn_idx=top1_nn_idx,
            mcluster_min=3,
        )

        c_raw = int(np.unique(roots).size)
        c_eff = int(np.unique(roots_eff).size)
        self.assertLess(c_eff, c_raw)
        self.assertGreater(c_eff, 2)
        self.assertEqual(size_eff.shape, roots.shape)


if __name__ == "__main__":
    unittest.main()
