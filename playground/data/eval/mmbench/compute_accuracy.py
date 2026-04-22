import json
import pandas as pd
import argparse



# Paths
pred_file = "playground/data/eval/mmbench/answers/mmbench_dev_20230712/llava-v1.5-13b.jsonl"
gt_file = "playground/data/eval/mmbench/mmbench_dev_20230712.tsv"

# Load predictions

def eval_model(args):

    pred_file = args.pred_file
    gt_file = args.gt_file
    experiment_name = args.experiment_name



    preds = {}
    with open(pred_file, "r") as f:
        for line in f:
            item = json.loads(line)
            preds[item["question_id"]] = item["text"].strip()

    # Load ground truth
    df = pd.read_csv(gt_file, sep="\t")

    correct = 0
    total = 0

    for _, row in df.iterrows():
        qid = row["index"]
        gt = row["answer"]  # MMBench uses 'answer' column

        if qid in preds:
            total += 1
            if preds[qid] == gt:
                correct += 1

    accuracy = correct / total

    print(f"{correct} | {total} | {accuracy:.4f} | {experiment_name}")

    return correct, total, accuracy, experiment_name



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, default=pred_file, help="Path to the predictions file (jsonl)")
    parser.add_argument("--gt-file", type=str, default=gt_file, help="Path to the ground truth file (tsv)")
    parser.add_argument("--experiment-name", type=str, default="mmbench_eval", help="Name of the evaluation experiment")
    args = parser.parse_args()

    

    eval_model(args)