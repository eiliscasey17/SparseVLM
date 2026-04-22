from PIL import Image
import torch
import matplotlib.pyplot as plt
from argparse import Namespace


import matplotlib.cm as cm
import matplotlib.patches as mpatches

import pandas as pd
import numpy as np





def register_key_hook(model, layer_idx):
    keys = {}

    def hook_fn(module, input, output):
        # output is already the projected key tensor
        keys["k"] = output.detach()

    handle = (
        model.vision_model
        .encoder.layers[layer_idx]
        .self_attn
        .k_proj
        .register_forward_hook(hook_fn)
    )

    return keys, handle

def cluster_keys(keys, num_clusters=4):
    cluster_size = keys.shape[1] // num_clusters
    avg_key = keys[:,1:,:].mean(dim=1,)

    cosine_sim = torch.nn.functional.cosine_similarity(avg_key.squeeze(0), keys.squeeze(0), dim=1)

    sorted_indices = torch.argsort(cosine_sim, descending=True).squeeze(0)
    clusters = [ sorted_indices[i:i+cluster_size] for i in range(0, len(sorted_indices), cluster_size) ]
   
    return clusters

def mean_cluster_attention(clusters, cls_attention):
    
    cluster_attentions = []
    for cluster in clusters:
        cluster_attention = cls_attention[:, -1, cluster.to(cls_attention.device)].squeeze(0).mean(dim=0)

        cluster_attentions.append(cluster_attention)
        
       
    return cluster_attentions


def generate_cluster_mask(attention,sparsity_budget,clusters, global_sparsity=.75):


    
    attn = attention[:, -1:, :]

    salience = attn.squeeze(0).squeeze(0)

    device = salience.device
    sparsity_budget = sparsity_budget.to(device)

    num_patches = salience.shape[-1]

    mask = torch.zeros(num_patches, dtype=torch.bool)

    sorted_indices = torch.argsort(sparsity_budget, descending=True)

    cluster_keys = {}

    global_keep = int(num_patches * (1 - global_sparsity))
    overflow = 0

    for cluster in sorted_indices:
        keep = global_keep * sparsity_budget[cluster] + overflow

        if keep > len(clusters[cluster]):
            overflow = keep - len(clusters[cluster])
            keep = len(clusters[cluster])
        else:
            overflow = 0
            keep = keep.int().item()

        cluster_keys[cluster.item()] = keep
        

    for i, cluster in enumerate(clusters):
        cluster = cluster.to(device)
        keep = cluster_keys[i]
        cluster_scores = salience[cluster.to(device)]
        topk_indices = torch.topk(cluster_scores, k=keep, largest=True).indices
        selected = cluster[topk_indices.to(device)]
        mask[selected] = True
    return mask

def prune_cluster(cls_attention, clusters, sparsity):
    cls_attention = cls_attention[:, -1, :].squeeze(0)
    cluster_salience = torch.tensor([cls_attention[cluster.to(cls_attention.device)].mean().item() for cluster in clusters])

    salience_mask = cluster_salience > cluster_salience.quantile(sparsity)
    indices_to_keep = []
    for i, cluster in enumerate(clusters):
        if salience_mask[i]:
            indices_to_keep.extend(cluster.cpu().numpy().tolist())
    
    return indices_to_keep

    
    

    




        

        


