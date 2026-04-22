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
    parser.add_argument("--output-file", type=str, default="sample_288_s85_attention_barplot.png")
    parser.add_argument("--layer", type=int, default=None,
                        help="Optional explicit layer. If omitted, uses the last layer in target_layers.")
    args = parser.parse_args()

    turns = load_sample_turns(args.input_file, args.sample_id)

    if len(turns) == 0:
        raise ValueError(f"No turn records found for sample {args.sample_id}")

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
    mean_salience = np.mean(np.stack(saliences, axis=0), axis=0)

    x = np.arange(len(mean_salience))

    common_x = [i for i in x if i in common_tokens]
    common_y = [mean_salience[i] for i in common_x]

    other_x = [i for i in x if i not in common_tokens]
    other_y = [mean_salience[i] for i in other_x]

    plt.figure(figsize=(16, 5))
    plt.bar(other_x, other_y, label="Other tokens")
    plt.bar(common_x, common_y, label="Common kept tokens")

    plt.xlabel("Token index")
    plt.ylabel("Attention value")
    plt.title(f"Sample {args.sample_id}, Sparsity 85, Layer {layer}: Token salience")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Saved plot to {args.output_file}")
    print(f"Number of common kept tokens: {len(common_tokens)}")
    print(f"Common token indices: {sorted(common_tokens)}")


if __name__ == "__main__":
    main()