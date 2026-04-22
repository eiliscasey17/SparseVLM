import json
import argparse
from collections import defaultdict


def load_records(file_path):
    by_sample = defaultdict(list)

    with open(file_path, "r") as f:
        for line in f:
            rec = json.loads(line)

            # skip summary lines
            if rec.get("summary_type") == "common_across_queries":
                continue

            if "turn_id" in rec and "layer_results" in rec:
                by_sample[rec["sample_id"]].append(rec)

    return by_sample


def compute_common_tokens(turns, layer):
    token_sets = []

    for rec in turns:
        layer_results = rec["layer_results"]

        if str(layer) in layer_results:
            kept = layer_results[str(layer)]["kept_token_indices"]
        elif layer in layer_results:
            kept = layer_results[layer]["kept_token_indices"]
        else:
            continue

        token_sets.append(set(kept))

    if len(token_sets) == 0:
        return []

    return sorted(list(set.intersection(*token_sets)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    args = parser.parse_args()

    by_sample = load_records(args.input_file)

    with open(args.output_file, "w") as out:
        for sample_id, turns in by_sample.items():

            target_layers = turns[0]["target_layers"]

            for layer in target_layers:
                common_tokens = compute_common_tokens(turns, layer)

                record = {
                    "sample_id": sample_id,
                    "layer": layer,
                    "common_token_indices": common_tokens
                }

                out.write(json.dumps(record) + "\n")

    print(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()