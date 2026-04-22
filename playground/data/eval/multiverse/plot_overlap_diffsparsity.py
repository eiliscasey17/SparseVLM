import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os


TOTAL_TOKENS = 576


def load_turns(jsonl_file):
    by_sample = defaultdict(list)

    with open(jsonl_file) as f:
        for line in f:
            rec = json.loads(line)

            if rec.get("summary_type") == "common_across_queries":
                continue

            if "sample_id" in rec and "turn_id" in rec:
                by_sample[rec["sample_id"]].append(rec)

    return by_sample


def get_last_layer_tokens(rec):
    layer = rec["target_layers"][-1]
    lr = rec["layer_results"]

    if str(layer) in lr:
        return set(lr[str(layer)]["kept_token_indices"])
    else:
        return set(lr[layer]["kept_token_indices"])


def compute_overlap(by_sample):
    overlap = {}

    for sample_id, turns in by_sample.items():
        token_sets = [get_last_layer_tokens(t) for t in turns]

        if not token_sets:
            continue

        common_tokens = set.intersection(*token_sets)
        overlap[sample_id] = len(common_tokens)

    return overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="sparsity_overlap_plot.png")
    args = parser.parse_args()

    sparsities = [75, 80, 85, 90, 95]

    overlap_data = {}
    sample_ids = None

    for s in sparsities:

        file_path = os.path.join(
            args.input_dir,
            f"sparsevila_keep_tokens_results_{s}.jsonl"
        )

        turns = load_turns(file_path)
        overlap = compute_overlap(turns)

        if sample_ids is None:
            sample_ids = sorted(overlap.keys(), key=lambda x: int(x))

        kept_tokens = int((1 - s/100) * TOTAL_TOKENS)

        overlap_data[s] = [
            overlap.get(sample, 0) / kept_tokens
            for sample in sample_ids
        ]

    x = np.arange(len(sample_ids))
    width = 0.15

    plt.figure(figsize=(14,6))

    for i, s in enumerate(sparsities):
        plt.bar(
            x + i*width,
            overlap_data[s],
            width,
            label=f"{s}% sparsity"
        )

    plt.xlabel("Sample ID")
    plt.ylabel("Overlap / Kept Tokens")
    plt.title("Overlap of visual tokens across conversation turns vs sparsity")
    plt.xticks(x + width*2, sample_ids, rotation=45)
    plt.legend(title="Sparsity")
    plt.tight_layout()

    plt.savefig(args.output, dpi=300)

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()