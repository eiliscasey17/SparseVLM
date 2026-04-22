import json
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt


def load_turn_data(jsonl_file):
    by_sample = defaultdict(list)

    with open(jsonl_file, "r") as f:
        for line in f:
            rec = json.loads(line)

            if rec.get("summary_type") == "common_across_queries":
                continue

            if "sample_id" in rec and "turn_id" in rec and "layer_results" in rec:
                by_sample[rec["sample_id"]].append(rec)

    return by_sample


def get_last_layer_kept(rec):
    last_layer = rec["target_layers"][-1]
    lr = rec["layer_results"]

    if str(last_layer) in lr:
        return set(lr[str(last_layer)]["kept_token_indices"])
    else:
        return set(lr[last_layer]["kept_token_indices"])


def compute_overlap_counts(by_sample):
    sample_ids = []
    overlap_counts = []

    for sample_id, turns in by_sample.items():

        token_sets = [get_last_layer_kept(r) for r in turns]

        if len(token_sets) == 0:
            continue

        common_tokens = set.intersection(*token_sets)

        sample_ids.append(str(sample_id))
        overlap_counts.append(len(common_tokens))

    return sample_ids, overlap_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="overlap_bar_plot.png")
    args = parser.parse_args()

    by_sample = load_turn_data(args.input_file)
    sample_ids, overlap_counts = compute_overlap_counts(by_sample)

    # sort samples numerically if possible
    try:
        order = sorted(range(len(sample_ids)), key=lambda i: int(sample_ids[i]))
        sample_ids = [sample_ids[i] for i in order]
        overlap_counts = [overlap_counts[i] for i in order]
    except:
        pass

    plt.figure(figsize=(12,6))
    plt.bar(sample_ids, overlap_counts)
    plt.xlabel("Sample ID")
    plt.ylabel("Number of overlapping tokens")
    plt.title("Visual tokens kept across all conversation turns")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(args.output_file, dpi=300)
    print(f"Saved plot to {args.output_file}")


if __name__ == "__main__":
    main()