# llava/model/vision_memory.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class VisionMemory:
    # full raw encoder tokens before projector
    full_tokens: torch.Tensor                 # [N, D]
    kept_original_indices: torch.Tensor       # [Nk]
    sink_original_indices: torch.Tensor       # [Ns]
    nonsink_original_indices: torch.Tensor    # [Nr]

    # clusters over non-sink kept tokens
    cluster_ids: torch.Tensor                 # [Nr] in [0, C-1]
    cluster_to_original: Dict[int, torch.Tensor]
    cluster_means: torch.Tensor               # [C, D]
    cluster_mins: torch.Tensor                # [C, D]
    cluster_maxs: torch.Tensor                # [C, D]

    # optional cached projected tokens
    projected_full_tokens: Optional[torch.Tensor] = None  # [N, Dp]

    # bookkeeping
    patch_grid: Optional[Tuple[int, int]] = None
    metadata: Optional[dict] = None


def ensure_vision_memory_device(vision_memory: VisionMemory, device: torch.device) -> VisionMemory:
    vision_memory.full_tokens = vision_memory.full_tokens.to(device)
    vision_memory.kept_original_indices = vision_memory.kept_original_indices.to(device)
    vision_memory.sink_original_indices = vision_memory.sink_original_indices.to(device)
    vision_memory.nonsink_original_indices = vision_memory.nonsink_original_indices.to(device)
    vision_memory.cluster_ids = vision_memory.cluster_ids.to(device)
    vision_memory.cluster_means = vision_memory.cluster_means.to(device)
    vision_memory.cluster_mins = vision_memory.cluster_mins.to(device)
    vision_memory.cluster_maxs = vision_memory.cluster_maxs.to(device)

    if vision_memory.projected_full_tokens is not None:
        vision_memory.projected_full_tokens = vision_memory.projected_full_tokens.to(device)

    vision_memory.cluster_to_original = {
        cid: original_indices.to(device)
        for cid, original_indices in vision_memory.cluster_to_original.items()
    }

    return vision_memory


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def compute_cls_salience(
    attentions: torch.Tensor,
    include_cls: bool = False,
) -> torch.Tensor:
    """
    attentions: [B, H, T, T] from a ViT layer
    Returns salience over tokens for batch size 1.
    Uses CLS -> token attention averaged across heads.
    """
    assert attentions.dim() == 4, f"Expected [B,H,T,T], got {attentions.shape}"
    assert attentions.size(0) == 1, "This scaffold expects batch size 1."

    cls_attn = attentions[0, :, 0, :]  # [H, T]
    salience = cls_attn.mean(dim=0)    # [T]

    if not include_cls:
        salience = salience[1:]        # patch tokens only

    return salience


def prune_tokens_by_salience(
    tokens: torch.Tensor,
    salience: torch.Tensor,
    prune_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    tokens: [N, D] patch tokens only
    salience: [N]
    Returns:
      kept_original_indices over patch-token indexing [0..N-1]
      kept_tokens [Nk, D]
    """
    assert tokens.dim() == 2
    assert salience.dim() == 1
    assert tokens.size(0) == salience.size(0)

    if prune_ratio <= 0:
        keep_mask = torch.ones_like(salience, dtype=torch.bool)
    else:
        threshold = torch.quantile(salience.float(), prune_ratio)
        keep_mask = salience >= threshold

    kept_original_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
    kept_tokens = tokens[kept_original_indices]

    return kept_original_indices, kept_tokens


def select_sink_tokens(
    kept_original_indices: torch.Tensor,
    salience: torch.Tensor,
    num_sink_tokens: int = 32,
    diversify_spatial: bool = False,
    patch_grid: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Returns sink token indices in ORIGINAL patch-token indexing.
    """
    kept_salience = salience[kept_original_indices]
    num_sink_tokens = min(num_sink_tokens, kept_original_indices.numel())

    if num_sink_tokens <= 0:
        return kept_original_indices[:0]

    # simple baseline: top salience among kept tokens
    top_local = torch.topk(kept_salience, k=num_sink_tokens, dim=0).indices
    sink_original_indices = kept_original_indices[top_local]

    # optional: sort for stable ordering
    sink_original_indices = torch.sort(sink_original_indices).values
    return sink_original_indices


def _make_spatial_features(
    num_tokens: int,
    patch_grid: Optional[Tuple[int, int]],
    device,
    dtype,
    original_indices: Optional[torch.Tensor] = None,
):
    if patch_grid is None:
        return None

    gh, gw = patch_grid
    total_tokens = gh * gw

    ys, xs = torch.meshgrid(
        torch.linspace(0, 1, gh, device=device, dtype=dtype),
        torch.linspace(0, 1, gw, device=device, dtype=dtype),
        indexing="ij"
    )
    full_xy = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [576, 2] for 24x24

    if original_indices is None:
        assert total_tokens == num_tokens, f"patch_grid={patch_grid} incompatible with {num_tokens} tokens"
        return full_xy

    return full_xy[original_indices]


def cluster_tokens_kmeans(
    tokens: torch.Tensor,
    num_clusters: int = 16,
    patch_grid: Optional[Tuple[int, int]] = None,
    original_indices: Optional[torch.Tensor] = None,
    spatial_weight: float = 0.05,
    num_iters: int = 20,
) -> torch.Tensor:
    """
    tokens: [N, D]
    returns cluster_ids: [N]
    """
    device = tokens.device
    dtype = tokens.dtype
    n, d = tokens.shape

    if n == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    num_clusters = max(1, min(num_clusters, n))

    feats = tokens
    xy = _make_spatial_features(
        n,
        patch_grid,
        device=device,
        dtype=dtype,
        original_indices=original_indices,
    )
    if xy is not None:
        feats = torch.cat([tokens, spatial_weight * xy], dim=-1)

    # init by random subset
    perm = torch.randperm(n, device=device)
    centers = feats[perm[:num_clusters]].clone()

    for _ in range(num_iters):
        dist = torch.cdist(feats.float(), centers.float())  # [N, C]
        cluster_ids = dist.argmin(dim=-1)

        new_centers = []
        for c in range(num_clusters):
            mask = cluster_ids == c
            if mask.any():
                new_centers.append(feats[mask].mean(dim=0))
            else:
                new_centers.append(centers[c])
        new_centers = torch.stack(new_centers, dim=0)

        if torch.allclose(new_centers, centers, atol=1e-4, rtol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    return cluster_ids


def cluster_tokens_semantic(
    tokens: torch.Tensor,
    num_clusters: int = 16,
    cluster_page_size: Optional[int] = None,
) -> torch.Tensor:
    """
    One-pass semantic clustering with balanced cluster sizes.

    1. Build one representative center by average pooling all tokens.
    2. Rank tokens by cosine similarity to that center.
    3. Partition the ranked tokens into equal-size (or as equal as possible) clusters.

    If ``cluster_page_size`` is provided, it is treated as the desired tokens per
    cluster and the number of clusters is derived from it. Otherwise ``num_clusters``
    is used directly.
    """
    device = tokens.device
    n, _ = tokens.shape

    if n == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    if cluster_page_size is not None and cluster_page_size > 0:
        num_clusters = max(1, (n + cluster_page_size - 1) // cluster_page_size)
    else:
        num_clusters = max(1, min(num_clusters, n))

    center = tokens.mean(dim=0, keepdim=True)  # [1, D]
    normalized_tokens = _safe_normalize(tokens.float(), dim=-1)
    normalized_center = _safe_normalize(center.float(), dim=-1)
    cosine_scores = torch.matmul(normalized_tokens, normalized_center.transpose(0, 1)).squeeze(-1)  # [N]

    sorted_indices = torch.argsort(cosine_scores, descending=True)

    # Balanced partitioning: cluster sizes differ by at most one token.
    ranked_cluster_ids = torch.div(
        torch.arange(n, device=device) * num_clusters,
        n,
        rounding_mode="floor",
    )
    ranked_cluster_ids = torch.clamp(ranked_cluster_ids, max=num_clusters - 1)

    cluster_ids = torch.empty(n, dtype=torch.long, device=device)
    cluster_ids[sorted_indices] = ranked_cluster_ids
    return cluster_ids


def compute_cluster_stats(
    nonsink_tokens: torch.Tensor,
    nonsink_original_indices: torch.Tensor,
    cluster_ids: torch.Tensor,
    num_clusters: int,
):
    """
    Build mean/min/max summaries per cluster.
    """
    if nonsink_tokens.size(0) == 0:
        d = nonsink_tokens.size(-1)
        return {}, torch.empty(0, d, device=nonsink_tokens.device), \
               torch.empty(0, d, device=nonsink_tokens.device), \
               torch.empty(0, d, device=nonsink_tokens.device)

    cluster_to_original = {}
    cluster_means = []
    cluster_mins = []
    cluster_maxs = []

    for c in range(num_clusters):
        mask = cluster_ids == c
        if not mask.any():
            continue

        toks = nonsink_tokens[mask]                       # [Nc, D]
        orig = nonsink_original_indices[mask]            # [Nc]

        cluster_to_original[c] = torch.sort(orig).values
        cluster_means.append(toks.mean(dim=0))
        cluster_mins.append(toks.min(dim=0).values)
        cluster_maxs.append(toks.max(dim=0).values)

    if len(cluster_means) == 0:
        d = nonsink_tokens.size(-1)
        return {}, torch.empty(0, d, device=nonsink_tokens.device), \
               torch.empty(0, d, device=nonsink_tokens.device), \
               torch.empty(0, d, device=nonsink_tokens.device)

    cluster_means = torch.stack(cluster_means, dim=0)
    cluster_mins = torch.stack(cluster_mins, dim=0)
    cluster_maxs = torch.stack(cluster_maxs, dim=0)

    return cluster_to_original, cluster_means, cluster_mins, cluster_maxs


def build_vision_memory(
    patch_tokens: torch.Tensor,              # [N, D], no CLS
    attentions: torch.Tensor,                # [B,H,T,T], last ViT layer
    prune_ratio: float = 0.2,
    num_sink_tokens: int = 32,
    num_clusters: int = 16,
    clustering_mode: str = "kmeans",
    cluster_page_size: Optional[int] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    spatial_weight: float = 0.05,
    metadata: Optional[dict] = None,
) -> VisionMemory:
    """
    patch_tokens are in ORIGINAL patch-token order [0..N-1].
    """
    salience = compute_cls_salience(attentions, include_cls=False)  # [N]

    kept_original_indices, kept_tokens = prune_tokens_by_salience(
        patch_tokens, salience, prune_ratio=prune_ratio
    )

    sink_original_indices = select_sink_tokens(
        kept_original_indices=kept_original_indices,
        salience=salience,
        num_sink_tokens=num_sink_tokens,
        patch_grid=patch_grid,
    )

    # remove sink tokens from clustering pool
    if sink_original_indices.numel() > 0:
        is_sink = torch.isin(kept_original_indices, sink_original_indices)
    else:
        is_sink = torch.zeros_like(kept_original_indices, dtype=torch.bool)

    nonsink_original_indices = kept_original_indices[~is_sink]
    nonsink_tokens = patch_tokens[nonsink_original_indices]

    if nonsink_tokens.size(0) > 0:
        if clustering_mode == "kmeans":
            cluster_ids = cluster_tokens_kmeans(
                nonsink_tokens,
                num_clusters=num_clusters,
                patch_grid=patch_grid,
                original_indices=nonsink_original_indices,
                spatial_weight=spatial_weight,
            )
        elif clustering_mode == "semantic":
            cluster_ids = cluster_tokens_semantic(
                nonsink_tokens,
                num_clusters=num_clusters,
                cluster_page_size=cluster_page_size,
            )
        else:
            raise ValueError(f"Unknown clustering_mode: {clustering_mode}")

        actual_num_clusters = int(cluster_ids.max().item()) + 1
        cluster_to_original, cluster_means, cluster_mins, cluster_maxs = compute_cluster_stats(
            nonsink_tokens=nonsink_tokens,
            nonsink_original_indices=nonsink_original_indices,
            cluster_ids=cluster_ids,
            num_clusters=actual_num_clusters,
        )
    else:
        cluster_ids = torch.empty(0, dtype=torch.long, device=patch_tokens.device)
        d = patch_tokens.size(-1)
        cluster_to_original = {}
        cluster_means = torch.empty(0, d, device=patch_tokens.device)
        cluster_mins = torch.empty(0, d, device=patch_tokens.device)
        cluster_maxs = torch.empty(0, d, device=patch_tokens.device)

    return VisionMemory(
        full_tokens=patch_tokens,
        kept_original_indices=kept_original_indices,
        sink_original_indices=sink_original_indices,
        nonsink_original_indices=nonsink_original_indices,
        cluster_ids=cluster_ids,
        cluster_to_original=cluster_to_original,
        cluster_means=cluster_means,
        cluster_mins=cluster_mins,
        cluster_maxs=cluster_maxs,
        projected_full_tokens=None,
        patch_grid=patch_grid,
        metadata={
            **(metadata or {}),
            "clustering_mode": clustering_mode,
            "cluster_page_size": cluster_page_size,
        },
    )

def rebuild_cluster_stats_from_projected_tokens(vision_memory):
    assert vision_memory.projected_full_tokens is not None, "projected_full_tokens not available"

    proj = vision_memory.projected_full_tokens
    vision_memory = ensure_vision_memory_device(vision_memory, proj.device)

    if vision_memory.nonsink_original_indices.numel() == 0:
        return vision_memory

    nonsink_proj = proj[vision_memory.nonsink_original_indices]
    cluster_ids = vision_memory.cluster_ids

    if cluster_ids.numel() == 0:
        d = proj.size(-1)
        vision_memory.cluster_means = torch.empty(0, d, device=proj.device, dtype=proj.dtype)
        vision_memory.cluster_mins = torch.empty(0, d, device=proj.device, dtype=proj.dtype)
        vision_memory.cluster_maxs = torch.empty(0, d, device=proj.device, dtype=proj.dtype)
        return vision_memory

    num_clusters = int(cluster_ids.max().item()) + 1

    cluster_means = []
    cluster_mins = []
    cluster_maxs = []

    for c in range(num_clusters):
        mask = cluster_ids == c
        if not mask.any():
            continue

        toks = nonsink_proj[mask]
        cluster_means.append(toks.mean(dim=0))
        cluster_mins.append(toks.min(dim=0).values)
        cluster_maxs.append(toks.max(dim=0).values)

    vision_memory.cluster_means = torch.stack(cluster_means, dim=0)
    vision_memory.cluster_mins = torch.stack(cluster_mins, dim=0)
    vision_memory.cluster_maxs = torch.stack(cluster_maxs, dim=0)

    return vision_memory
