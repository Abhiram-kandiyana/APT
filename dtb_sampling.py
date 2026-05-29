"""
DTB scoring utilities for APT.

Implements the core equations from APT-DTB formal draft:
- Eq. (3): local density proxy rho_i from k_rho nearest-neighbor distances
- Eq. (4): local threshold radius t_i from the k_t-th neighbor distance
- Eq. (5)-(6): steepest-uphill parent pointer pi(i)
- Eq. (7): boundary score b(i) used for ranking candidates
"""

import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors


# Reuse the same image-embedding model instance across rounds to avoid repeated model loads.
_CLIP_EMBEDDER_CACHE: Dict[Tuple[str, str], "CLIPImageEmbedder"] = {}


class CLIPImageEmbedder:
    def __init__(self, model_name: str, device: str = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.preprocess = None
        self.torch = None
        self.loaded_model_name = None
        self.backend = None

    def _ensure_loaded(self):
        # open_clip backends expose `preprocess` (not HF `processor`), while
        # transformers backends expose `processor`.
        if self.model is not None and (self.processor is not None or self.preprocess is not None):
            return

        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            try:
                from transformers import AutoImageProcessor
            except Exception:
                AutoImageProcessor = None
        except ImportError as e:
            raise ImportError(
                "DTB selection requires torch and transformers. "
                "Install them to use --selection_method dtb."
            ) from e

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_name_lc = str(self.model_name).strip().lower()

        # BiomedCLIP is most reliable via open_clip + HF hub weights.
        if model_name_lc == "microsoft/biomedclip-pubmedbert_256-vit_base_patch16_224":
            try:
                import open_clip
            except ImportError as e:
                raise ImportError(
                    "BiomedCLIP requires 'open_clip_torch'. Install with: "
                    "python -m pip install open_clip_torch"
                ) from e
            hf_id = f"hf-hub:{self.model_name}"
            created = open_clip.create_model_from_pretrained(hf_id, device=self.device)
            if isinstance(created, tuple):
                self.model = created[0]
                # open_clip variants may return (model, preprocess) or longer tuples.
                self.preprocess = created[1] if len(created) > 1 else None
                if self.preprocess is None and len(created) > 2:
                    self.preprocess = created[2]
            else:
                self.model = created
                self.preprocess = None
            if self.preprocess is None:
                raise RuntimeError(
                    "open_clip did not return an image preprocessing transform for BiomedCLIP."
                )
            self.loaded_model_name = self.model_name
            self.backend = "open_clip"
            self.model.eval()
            self.torch = torch
            return

        # Phikon-v2 requires CLS token extraction from last_hidden_state.
        if model_name_lc == "owkin/phikon-v2":
            processor_cls = AutoImageProcessor if AutoImageProcessor is not None else AutoProcessor
            self.processor = processor_cls.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(self.device)
            self.loaded_model_name = self.model_name
            self.backend = "phikonv2"
            self.model.eval()
            self.torch = torch
            return

        # Do not silently switch to a different embedding model on load failure.
        # If a model is requested explicitly, fail fast so experiment semantics remain correct.
        candidates = [self.model_name]

        load_errors = []
        for candidate in candidates:
            candidate_lc = str(candidate).strip().lower()
            is_siglip_family = "siglip" in candidate_lc
            try:
                # Prefer generic auto classes for compatibility with BiomedCLIP variants.
                self.processor = AutoProcessor.from_pretrained(candidate, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(candidate, trust_remote_code=True).to(self.device)
                self.loaded_model_name = candidate
                self.backend = "transformers"
                break
            except Exception as auto_err:
                if is_siglip_family:
                    # Do not force SigLIP checkpoints through CLIPModel; incompatible architectures.
                    load_errors.append(f"{candidate} -> Auto load error: {auto_err}")
                    continue
                try:
                    from transformers import CLIPImageProcessor, CLIPModel

                    # Fallback for standard CLIP checkpoints.
                    self.processor = CLIPImageProcessor.from_pretrained(candidate)
                    self.model = CLIPModel.from_pretrained(candidate).to(self.device)
                    self.loaded_model_name = candidate
                    self.backend = "transformers"
                    break
                except Exception as clip_err:
                    load_errors.append(
                        f"{candidate} -> Auto load error: {auto_err}; CLIP load error: {clip_err}"
                    )

        if self.model is None or self.processor is None:
            joined = " | ".join(load_errors) if load_errors else "unknown error"
            raise RuntimeError(
                "Failed to load any DTB embedding model. "
                f"Tried: {candidates}. Details: {joined}"
            )

        self.model.eval()
        self.torch = torch

    def embed_image_paths(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        self._ensure_loaded()

        all_embeddings = []
        with self.torch.no_grad():
            for start in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[start:start + batch_size]
                images = []
                for image_path in batch_paths:
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image not found for DTB embedding: {image_path}")
                    with Image.open(image_path) as img:
                        images.append(img.convert("RGB"))

                if self.backend == "open_clip":
                    image_tensors = [self.preprocess(img) for img in images]
                    image_batch = self.torch.stack(image_tensors).to(self.device)
                    image_features = self.model.encode_image(image_batch)
                elif self.backend == "phikonv2":
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    if hasattr(outputs, "last_hidden_state"):
                        image_features = outputs.last_hidden_state[:, 0, :]
                    elif hasattr(outputs, "pooler_output"):
                        image_features = outputs.pooler_output
                    else:
                        raise RuntimeError(
                            "Unsupported embedding output format from model: owkin/phikon-v2"
                        )
                else:
                    inputs = self.processor(images=images, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    if hasattr(self.model, "get_image_features"):
                        image_features = self.model.get_image_features(**inputs)
                    else:
                        outputs = self.model(**inputs)
                        if hasattr(outputs, "image_embeds"):
                            image_features = outputs.image_embeds
                        elif hasattr(outputs, "pooler_output"):
                            image_features = outputs.pooler_output
                        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
                            image_features = outputs[0]
                        else:
                            raise RuntimeError(
                                f"Unsupported embedding output format from model: {self.model_name}"
                            )
                image_features = image_features / (image_features.norm(dim=-1, keepdim=True) + 1e-12)
                all_embeddings.append(image_features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def _get_clip_embedder(model_name: str, device: str = None) -> CLIPImageEmbedder:
    key = (model_name, device or "auto")
    if key not in _CLIP_EMBEDDER_CACHE:
        _CLIP_EMBEDDER_CACHE[key] = CLIPImageEmbedder(model_name=model_name, device=device)
    return _CLIP_EMBEDDER_CACHE[key]


def _trace_roots(parents: np.ndarray) -> np.ndarray:
    n = len(parents)
    roots = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        if roots[i] != -1:
            continue

        path = []
        seen = set()
        cur = i
        while True:
            if roots[cur] != -1:
                root = roots[cur]
                break
            if cur in seen:
                # Defensive fallback; cycles are not expected due to strict uphill condition.
                root = cur
                break
            seen.add(cur)
            path.append(cur)
            parent = int(parents[cur])
            if parent == -1:
                root = cur
                break
            cur = parent

        for node in path:
            roots[node] = root

    return roots


def _compute_effective_basin_ids(
    roots: np.ndarray,
    parents: np.ndarray,
    top1_nn_idx: np.ndarray,
    mcluster_min: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build effective basin IDs by squashing tiny basins into nearby non-tiny basins.

    Strategy (C1):
    - Keep original basin id when basin size >= mcluster_min.
    - For tiny basins, follow parent chain first.
    - If that cannot reach a non-tiny basin, follow nearest-neighbor chain as a fallback.
    - If no non-tiny basin is found, keep original basin id.
    """
    roots = np.asarray(roots, dtype=np.int32)
    parents = np.asarray(parents, dtype=np.int32)
    top1_nn_idx = np.asarray(top1_nn_idx, dtype=np.int32)
    n = int(roots.shape[0])
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    unique_roots, root_counts = np.unique(roots, return_counts=True)
    root_count_map = {int(r): int(c) for r, c in zip(unique_roots.tolist(), root_counts.tolist())}

    mmin = int(max(1, int(mcluster_min)))
    eff = roots.copy()

    for i in range(n):
        raw_root = int(roots[i])
        raw_size = int(root_count_map.get(raw_root, 0))
        if raw_size >= mmin:
            eff[i] = raw_root
            continue

        resolved = None
        cur = int(i)
        visited = set()
        hops = 0
        max_hops = max(1, n + 5)

        # First: strict parent-chain traversal.
        while 0 <= cur < n and cur not in visited and hops < max_hops:
            visited.add(cur)
            cur_root = int(roots[cur])
            cur_size = int(root_count_map.get(cur_root, 0))
            if cur_size >= mmin:
                resolved = cur_root
                break
            parent = int(parents[cur]) if cur < parents.size else -1
            if parent < 0:
                break
            cur = parent
            hops += 1

        # Fallback: nearest-neighbor chain to avoid giant+dust pathologies.
        if resolved is None:
            cur = int(i)
            visited = set()
            hops = 0
            while 0 <= cur < n and cur not in visited and hops < max_hops:
                visited.add(cur)
                cur_root = int(roots[cur])
                cur_size = int(root_count_map.get(cur_root, 0))
                if cur_size >= mmin:
                    resolved = cur_root
                    break
                nxt = int(top1_nn_idx[cur]) if cur < top1_nn_idx.size else -1
                if nxt < 0 or nxt == cur:
                    break
                cur = nxt
                hops += 1

        eff[i] = int(resolved) if resolved is not None else raw_root

    unique_eff, counts_eff = np.unique(eff, return_counts=True)
    eff_count_map = {int(r): int(c) for r, c in zip(unique_eff.tolist(), counts_eff.tolist())}
    eff_sizes = np.asarray([eff_count_map.get(int(r), 0) for r in eff.tolist()], dtype=np.int32)
    return eff.astype(np.int32), eff_sizes


def _load_embedding_cache(cache_path: str, image_paths: List[str]) -> np.ndarray:
    if not cache_path or not os.path.exists(cache_path):
        return None

    try:
        with np.load(cache_path, allow_pickle=True) as data:
            cached_paths = [str(p) for p in data["image_paths"].tolist()]
            if cached_paths != [str(p) for p in image_paths]:
                return None
            embeddings = data["embeddings"]
            if embeddings.shape[0] != len(image_paths):
                return None
            return embeddings
    except Exception:
        return None


def _save_embedding_cache(cache_path: str, image_paths: List[str], embeddings: np.ndarray):
    if not cache_path:
        return

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    np.savez_compressed(
        cache_path,
        image_paths=np.array([str(p) for p in image_paths], dtype=object),
        embeddings=embeddings,
    )


def score_candidates_with_dtb(
    image_paths: List[str],
    k: int = 80,
    k_rho: int = 30,
    k_t: int = 30,
    k_b: int = 20,
    use_mutual_knn: bool = False,
    clip_model_name: str = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    clip_batch_size: int = 32,
    clip_device: str = None,
    embedding_cache_path: str = None,
    mcluster_min: int = 20,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute DTB boundary scores for a candidate pool.

    Returns:
        - boundary_scores: score used for active-set ranking (higher = more boundary-like)
        - meta: per-sample diagnostics (rho, root id, cluster size)
    """
    n = len(image_paths)
    if n == 0:
        return np.array([], dtype=float), {
            "rho": np.array([], dtype=float),
            "root": np.array([], dtype=int),
            "cluster_size": np.array([], dtype=int),
            "root_eff": np.array([], dtype=int),
            "cluster_size_eff": np.array([], dtype=int),
            "parent": np.array([], dtype=np.int32),
            "embeddings": np.array([], dtype=np.float32),
            "knn_indices": np.array([], dtype=int),
            "knn_distances": np.array([], dtype=float),
            "mutual_mask": np.array([], dtype=bool),
            "mutual_degree": np.array([], dtype=int),
            "first_nn_distance": np.array([], dtype=float),
            "top1_nn_idx": np.array([], dtype=np.int32),
        }

    if n == 1:
        return np.array([0.0], dtype=float), {
            "rho": np.array([0.0], dtype=float),
            "root": np.array([0], dtype=int),
            "cluster_size": np.array([1], dtype=int),
            "root_eff": np.array([0], dtype=int),
            "cluster_size_eff": np.array([1], dtype=int),
            "parent": np.array([-1], dtype=np.int32),
            "embeddings": np.zeros((1, 0), dtype=np.float32),
            "knn_indices": np.zeros((1, 0), dtype=np.int32),
            "knn_distances": np.zeros((1, 0), dtype=np.float32),
            "mutual_mask": np.zeros((1, 0), dtype=bool),
            "mutual_degree": np.array([0], dtype=np.int32),
            "first_nn_distance": np.array([0.0], dtype=np.float32),
            "top1_nn_idx": np.array([-1], dtype=np.int32),
        }

    # Cache is round-specific and keyed by ordered candidate image paths.
    embeddings = _load_embedding_cache(embedding_cache_path, image_paths)
    if embeddings is None:
        embedder = _get_clip_embedder(model_name=clip_model_name, device=clip_device)
        embeddings = embedder.embed_image_paths(image_paths=image_paths, batch_size=clip_batch_size)
        _save_embedding_cache(embedding_cache_path, image_paths, embeddings)

    # Guard all neighbor counts so they remain valid for small candidate pools.
    max_k = min(max(k, k_rho, k_t, k_b), n - 1)
    k_graph = min(k, max_k)
    k_rho_eff = min(k_rho, max_k)
    k_t_eff = min(k_t, k_graph)
    k_b_eff = min(k_b, max_k)

    neighbors = NearestNeighbors(n_neighbors=max_k + 1, metric="cosine")
    neighbors.fit(embeddings)
    distances_full, indices_full = neighbors.kneighbors(embeddings)

    # Remove self-neighbor from each row so all slices are true neighbors.
    distances = distances_full[:, 1:]
    indices = indices_full[:, 1:]

    graph_distances = distances[:, :k_graph]
    graph_indices = indices[:, :k_graph]

    # Precompute mutual kNN mask if requested.
    # An edge i->j is mutual if j is in graph_indices[i] AND i is in graph_indices[j].
    mutual_mask = np.ones((n, k_graph), dtype=bool)
    if use_mutual_knn:
        # Create adjacency matrix for efficient O(1) checks.
        adj = np.zeros((n, n), dtype=bool)
        rows = np.repeat(np.arange(n), k_graph)
        cols = graph_indices.flatten()
        adj[rows, cols] = True
        
        # Check reverse edges
        for i in range(n):
            for j_idx, neighbor in enumerate(graph_indices[i]):
                if not adj[neighbor, i]:
                    mutual_mask[i, j_idx] = False

    # Eq. (3): rank-based local density proxy.
    rho_distances = distances[:, :k_rho_eff]
    rho = 1.0 / (np.mean(rho_distances, axis=1) + 1e-12)
    
    parents = np.full(n, -1, dtype=np.int32)

    for i in range(n):
        if use_mutual_knn:
            mutual_row_mask = mutual_mask[i]
            cand_neighbors = graph_indices[i][mutual_row_mask]
            cand_distances = graph_distances[i][mutual_row_mask]
        else:
            cand_neighbors = graph_indices[i]
            cand_distances = graph_distances[i]

        if cand_distances.size == 0:
            continue

        # Eq. (4): locality radius from the k_t-th nearest neighbor in standard kNN.
        threshold_radius = graph_distances[i, k_t_eff - 1]

        # Eq. (5): only local neighbors with strictly higher density can be parents.
        mask = (cand_distances <= threshold_radius) & (rho[cand_neighbors] > rho[i])

        if not np.any(mask):
            continue

        uphill_neighbors = cand_neighbors[mask]
        uphill_distances = cand_distances[mask]
        # Eq. (6): choose the steepest uphill transition.
        gain = (rho[uphill_neighbors] - rho[i]) / (uphill_distances + 1e-12)
        parents[i] = int(uphill_neighbors[np.argmax(gain)])

    roots = _trace_roots(parents)

    unique_roots, root_counts = np.unique(roots, return_counts=True)
    root_count_map = {int(root): int(count) for root, count in zip(unique_roots, root_counts)}
    cluster_sizes = np.array([root_count_map[int(root)] for root in roots], dtype=np.int32)
    top1_nn_idx = indices[:, 0].astype(np.int32)
    roots_eff, cluster_sizes_eff = _compute_effective_basin_ids(
        roots=roots,
        parents=parents,
        top1_nn_idx=top1_nn_idx,
        mcluster_min=int(mcluster_min),
    )

    # Eq. (7): boundary score is high when local neighbors frequently belong to other roots.
    deg_mut_all = np.sum(mutual_mask, axis=1).astype(np.int32)
    if use_mutual_knn:
        boundary_scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            mutual_neighbors = graph_indices[i][mutual_mask[i]]
            boundary_neighbors = mutual_neighbors[:k_b_eff]
            if boundary_neighbors.size == 0:
                boundary_scores[i] = 0.0
                continue
            same_cluster_count = np.sum(roots[boundary_neighbors] == roots[i], dtype=np.float32)
            boundary_scores[i] = 1.0 - (same_cluster_count / (boundary_neighbors.size + 1e-12))
    else:
        boundary_neighbors = indices[:, :k_b_eff]
        same_cluster = roots[boundary_neighbors] == roots[:, None]
        boundary_scores = 1.0 - (
            np.sum(same_cluster.astype(np.float32), axis=1) / (k_b_eff + 1e-12)
        )

    meta = {
        "rho": rho.astype(np.float32),
        "root": roots.astype(np.int32),
        "cluster_size": cluster_sizes.astype(np.int32),
        "root_eff": roots_eff.astype(np.int32),
        "cluster_size_eff": cluster_sizes_eff.astype(np.int32),
        "parent": parents.astype(np.int32),
        "embeddings": embeddings.astype(np.float32),
        "knn_indices": graph_indices.astype(np.int32),
        "knn_distances": graph_distances.astype(np.float32),
        "mutual_mask": mutual_mask.astype(bool),
        "mutual_degree": deg_mut_all.astype(np.int32),
        "first_nn_distance": distances[:, 0].astype(np.float32),
        "top1_nn_idx": top1_nn_idx,
    }
    return boundary_scores.astype(np.float32), meta
