from PIL import Image
import torch
import matplotlib.pyplot as plt
from argparse import Namespace


import pandas as pd
import numpy as np



def generate_mask(cls_attention, sparsity = .75):
    saliance = cls_attention.squeeze(0).squeeze(0)
    saliance_mask = saliance > saliance.quantile(sparsity)
    return saliance_mask


def generate_query_aware_mask(attention_scores, sparsity=.75):
    if attention_scores.numel() == 0 or sparsity <= 0:
        return torch.ones_like(attention_scores, dtype=torch.bool)

    if sparsity >= 1:
        keep_mask = torch.zeros_like(attention_scores, dtype=torch.bool)
        keep_mask[torch.argmax(attention_scores)] = True
        return keep_mask

    threshold = torch.quantile(attention_scores.float(), sparsity)
    keep_mask = attention_scores > threshold

    if not torch.any(keep_mask):
        keep_mask[torch.argmax(attention_scores)] = True

    return keep_mask


def get_sink_tokens(cls_attention, sparsity = .75):

    count = torch.zeros(cls_attention.shape[2], dtype=torch.float32)

    for head in range(cls_attention.shape[1]):
        saliance = cls_attention[:, head, :].squeeze(0)
        saliance_mask = saliance > saliance.quantile(sparsity)
        for idx, mask in enumerate(saliance_mask):
            if mask:
                count[idx] += 1

    saliance_mask = count > count.quantile(sparsity)
    return saliance_mask



def compare_masks(mask1, mask2):
    intersection = (mask1 & mask2).sum().item()
    union = (mask1 | mask2).sum().item()
    jaccard_similarity = intersection / union if union != 0 else 0.0
    return jaccard_similarity




