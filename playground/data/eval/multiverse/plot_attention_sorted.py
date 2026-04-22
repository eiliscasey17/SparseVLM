import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_sample_turns(jsonl_file, sample_id):
    turns = []
    with open(jsonl_file, "r") as f:
        for line in f:
            rec = json.loads(line)

            if rec.get("summary_type") == "common_across_queries":
                continue

            if str(rec.get("sample_id")) == str(sample_id):
                turns.append(rec)

    turns = sorted(turns, key=lambda x: x["turn_id"])
    return turns


def get_layer_data(rec, layer):
    lr = rec["layer_results"]
    if str(layer) in lr:
        return lr[str(layer)]
    return lr[layer]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, default="sparsevila_keep_tokens_results_85.jsonl")
    parser.add_argument("--sample-id", type=str, default="288")
    parser.add_argument("--output-file", type=str, default="sample_288_sorted_attention.png")
    parser.add_argument("--layer", type=int, default=None)
    args = parser.parse_args()

    turns = load_sample_turns(args.input_file, args.sample_id)

    if len(turns) == 0:
        raise ValueError("Sample not found")

    if args.layer is None:
        layer = turns[0]["target_layers"][-1]
    else:
        layer = args.layer

    kept_sets = []
    saliences = []

    for rec in turns:
        layer_data = get_layer_data(rec, layer)

        kept_sets.append(set(layer_data["kept_token_indices"]))
        saliences.append(np.array(layer_data["salience"], dtype=float))

    common_tokens = set.intersection(*kept_sets)

    # average salience across turns
    mean_salience = np.mean(np.stack(saliences), axis=0)

    token_indices = np.arange(len(mean_salience))

    # sort tokens by salience
    sorted_order = np.argsort(-mean_salience)

    sorted_salience = mean_salience[sorted_order]
    sorted_tokens = token_indices[sorted_order]

    colors = [
        "tab:red" if token in common_tokens else "tab:blue"
        for token in sorted_tokens
    ]

    plt.figure(figsize=(16,5))

    plt.bar(
        np.arange(len(sorted_salience)),
        sorted_salience,
        color=colors
    )

    plt.xlabel("Tokens ranked by attention value")
    plt.ylabel("Attention value")
    plt.title(f"Sample {args.sample_id} | Sparsity 85 | Layer {layer}")

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)

    print(f"Saved plot to {args.output_file}")
    print(f"Common tokens: {sorted(common_tokens)}")


if __name__ == "__main__":
    main()