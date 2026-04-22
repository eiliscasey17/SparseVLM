# llava/model/vision_retriever.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .vision_memory import VisionMemory, ensure_vision_memory_device


@dataclass
class VisionRetrievalConfig:
    prune_ratio: float = 0.2
    num_sink_tokens: int = 32
    num_clusters: int = 16
    clustering_mode: str = "kmeans"          # ["kmeans", "semantic"]
    cluster_page_size: Optional[int] = None  # semantic mode: desired tokens per cluster/page
    topk_clusters: int = 4
    retrieval_mode: str = "quest_upper_bound"   # ["quest_upper_bound", "cosine_mean"]
    token_budget: Optional[int] = None          # optional cap on total selected visual tokens
    use_projected_tokens_for_output: bool = False


def build_text_query_representation(
    embed_tokens_module,
    input_ids: torch.Tensor,
    image_token_index: int,
) -> torch.Tensor:
    """
    input_ids: [1, L]
    Pools only text tokens, ignoring image placeholder token.
    Returns [D]
    """
    assert input_ids.size(0) == 1, "This scaffold expects batch size 1."

    text_mask = input_ids[0] != image_token_index
    text_ids = input_ids[:, text_mask]  # [1, Lt]

    if text_ids.numel() == 0:
        # fallback
        text_ids = input_ids[:, :1]

    text_embeds = embed_tokens_module(text_ids)  # [1, Lt, D]
    query = text_embeds.mean(dim=1).squeeze(0)   # [D]
    return query


def score_clusters_quest_upper_bound(
    query: torch.Tensor,         # [D]
    cluster_mins: torch.Tensor,  # [C, D]
    cluster_maxs: torch.Tensor,  # [C, D]
) -> torch.Tensor:
    """
    QUEST-style upper bound:
    for each dimension, choose max contribution depending on sign(query_d).
    """
    if cluster_mins.size(0) == 0:
        return torch.empty(0, device=query.device, dtype=query.dtype)

    query = query.to(cluster_mins.device, dtype=cluster_mins.dtype)
    cluster_maxs = cluster_maxs.to(cluster_mins.device, dtype=cluster_mins.dtype)
    q = query.unsqueeze(0)  # [1, D]
    per_dim_best = torch.where(q >= 0, q * cluster_maxs, q * cluster_mins)  # [C, D]
    scores = per_dim_best.sum(dim=-1)  # [C]
    return scores


def score_clusters_cosine_mean(
    query: torch.Tensor,
    cluster_means: torch.Tensor,
) -> torch.Tensor:
    if cluster_means.size(0) == 0:
        return torch.empty(0, device=query.device, dtype=query.dtype)

    query = query.to(cluster_means.device, dtype=cluster_means.dtype)
    q = F.normalize(query, dim=-1)
    cm = F.normalize(cluster_means, dim=-1)
    return torch.matmul(cm, q)  # [C]


def retrieve_topk_clusters(
    vision_memory: VisionMemory,
    query: torch.Tensor,
    topk_clusters: int = 4,
    retrieval_mode: str = "quest_upper_bound",
):
    num_clusters = vision_memory.cluster_means.size(0)
    if num_clusters == 0:
        return torch.empty(0, dtype=torch.long, device=query.device), torch.empty(0, device=query.device)

    if retrieval_mode == "quest_upper_bound":
        scores = score_clusters_quest_upper_bound(
            query=query,
            cluster_mins=vision_memory.cluster_mins,
            cluster_maxs=vision_memory.cluster_maxs,
        )
    elif retrieval_mode == "cosine_mean":
        scores = score_clusters_cosine_mean(
            query=query,
            cluster_means=vision_memory.cluster_means,
        )
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    k = min(topk_clusters, scores.numel())
    vals, idx = torch.topk(scores, k=k, dim=0)
    return idx, vals


def rank_clusters(
    vision_memory: VisionMemory,
    query: torch.Tensor,
    retrieval_mode: str = "quest_upper_bound",
):
    num_clusters = vision_memory.cluster_means.size(0)
    if num_clusters == 0:
        return (
            torch.empty(0, dtype=torch.long, device=query.device),
            torch.empty(0, device=query.device),
        )

    if retrieval_mode == "quest_upper_bound":
        scores = score_clusters_quest_upper_bound(
            query=query,
            cluster_mins=vision_memory.cluster_mins,
            cluster_maxs=vision_memory.cluster_maxs,
        )
    elif retrieval_mode == "cosine_mean":
        scores = score_clusters_cosine_mean(
            query=query,
            cluster_means=vision_memory.cluster_means,
        )
    else:
        raise ValueError(f"Unknown retrieval_mode: {retrieval_mode}")

    vals, idx = torch.sort(scores, descending=True, dim=0)
    return idx, vals


def gather_selected_original_indices(
    vision_memory: VisionMemory,
    selected_cluster_ids: torch.Tensor,
    token_budget: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns ORIGINAL patch-token indices, sorted in original order.
    """
    vision_memory = ensure_vision_memory_device(vision_memory, vision_memory.full_tokens.device)

    selected = []

    if vision_memory.sink_original_indices.numel() > 0:
        selected.append(vision_memory.sink_original_indices)

    for cid in selected_cluster_ids.tolist():
        if cid in vision_memory.cluster_to_original:
            selected.append(vision_memory.cluster_to_original[cid])

    if len(selected) == 0:
        return torch.empty(0, dtype=torch.long, device=vision_memory.full_tokens.device)

    selected = torch.cat(selected, dim=0)
    selected = torch.unique(selected, sorted=True)

    if token_budget is not None and selected.numel() > token_budget:
        # always keep sinks first; then fill with lowest original index remainder
        sinks = vision_memory.sink_original_indices
        sinks = torch.unique(sinks, sorted=True)

        remaining = selected[~torch.isin(selected, sinks)]
        keep_extra = max(0, token_budget - sinks.numel())
        selected = torch.cat([sinks, remaining[:keep_extra]], dim=0)
        selected = torch.unique(selected, sorted=True)

    return selected


def gather_selected_original_indices_to_budget(
    vision_memory: VisionMemory,
    ranked_cluster_ids: torch.Tensor,
    token_budget: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns ORIGINAL patch-token indices, sorted in original order, and the
    cluster ids actually used to build that set.
    """
    vision_memory = ensure_vision_memory_device(vision_memory, vision_memory.full_tokens.device)
    device = vision_memory.full_tokens.device

    sinks = torch.unique(vision_memory.sink_original_indices, sorted=True)
    if token_budget is not None and token_budget <= sinks.numel():
        return sinks[:token_budget], torch.empty(0, dtype=torch.long, device=device)

    selected = []
    used_cluster_ids = []

    if sinks.numel() > 0:
        selected.append(sinks)

    current_count = int(sinks.numel())
    remaining_budget = None if token_budget is None else max(0, token_budget - current_count)

    for cid in ranked_cluster_ids.tolist():
        if cid not in vision_memory.cluster_to_original:
            continue

        cluster_tokens = vision_memory.cluster_to_original[cid]
        if cluster_tokens.numel() == 0:
            continue

        if remaining_budget is None:
            selected.append(cluster_tokens)
            used_cluster_ids.append(cid)
            continue

        if remaining_budget <= 0:
            break

        take_tokens = cluster_tokens[:remaining_budget]
        if take_tokens.numel() > 0:
            selected.append(take_tokens)
            used_cluster_ids.append(cid)
            remaining_budget -= int(take_tokens.numel())

        if remaining_budget <= 0:
            break

    if len(selected) == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

    selected_original_indices = torch.cat(selected, dim=0)
    selected_original_indices = torch.unique(selected_original_indices, sorted=True)
    used_cluster_ids = torch.tensor(used_cluster_ids, dtype=torch.long, device=device)
    return selected_original_indices, used_cluster_ids


def gather_selected_tokens(
    vision_memory: VisionMemory,
    selected_original_indices: torch.Tensor,
    use_projected_tokens: bool = False,
) -> torch.Tensor:
    target_device = vision_memory.projected_full_tokens.device if use_projected_tokens else vision_memory.full_tokens.device
    vision_memory = ensure_vision_memory_device(vision_memory, target_device)
    selected_original_indices = selected_original_indices.to(target_device)

    if use_projected_tokens:
        assert vision_memory.projected_full_tokens is not None, "projected_full_tokens not cached"
        return vision_memory.projected_full_tokens[selected_original_indices]
    return vision_memory.full_tokens[selected_original_indices]


def retrieve_visual_tokens_for_turn(
    vision_memory: VisionMemory,
    query: torch.Tensor,
    topk_clusters: int = 4,
    retrieval_mode: str = "quest_upper_bound",
    token_budget: Optional[int] = None,
    use_projected_tokens: bool = False,
):
    ranked_cluster_ids, ranked_cluster_scores = rank_clusters(
        vision_memory=vision_memory,
        query=query,
        retrieval_mode=retrieval_mode,
    )

    if token_budget is not None:
        selected_original_indices, cluster_ids = gather_selected_original_indices_to_budget(
            vision_memory=vision_memory,
            ranked_cluster_ids=ranked_cluster_ids,
            token_budget=token_budget,
        )

        if cluster_ids.numel() > 0:
            selected_mask = torch.isin(ranked_cluster_ids, cluster_ids)
            cluster_scores = ranked_cluster_scores[selected_mask]
        else:
            cluster_scores = torch.empty(0, device=query.device, dtype=query.dtype)
    else:
        cluster_ids = ranked_cluster_ids[:min(topk_clusters, ranked_cluster_ids.numel())]
        cluster_scores = ranked_cluster_scores[:cluster_ids.numel()]
        selected_original_indices = gather_selected_original_indices(
            vision_memory=vision_memory,
            selected_cluster_ids=cluster_ids,
            token_budget=None,
        )

    selected_tokens = gather_selected_tokens(
        vision_memory=vision_memory,
        selected_original_indices=selected_original_indices,
        use_projected_tokens=use_projected_tokens,
    )

    info = {
        "selected_cluster_ids": cluster_ids.detach().cpu().tolist(),
        "cluster_scores": cluster_scores.detach().float().cpu().tolist(),
        "selected_original_indices": selected_original_indices.detach().cpu().tolist(),
        "sink_original_indices": vision_memory.sink_original_indices.detach().cpu().tolist(),
    }
    return selected_tokens, info
