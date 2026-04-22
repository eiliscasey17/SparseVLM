import os
import json
import argparse
import importlib.util
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--upload-dir", type=str, required=True)
    parser.add_argument("--experiment", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)

    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in open(os.path.join(args.result_dir, f"{args.experiment}.jsonl")):
        pred = json.loads(pred)
        cur_df.loc[df['index'] == pred['question_id'], 'prediction'] = pred['text']

    xlsx_path = os.path.join(args.upload_dir, f"{args.experiment}.xlsx")
    csv_path = os.path.join(args.upload_dir, f"{args.experiment}.csv")

    if importlib.util.find_spec("openpyxl") is not None:
        cur_df.to_excel(xlsx_path, index=False, engine="openpyxl")
        print(f"Saved submission file to {xlsx_path}")
    elif importlib.util.find_spec("xlsxwriter") is not None:
        cur_df.to_excel(xlsx_path, index=False, engine="xlsxwriter")
        print(f"Saved submission file to {xlsx_path}")
    else:
        cur_df.to_csv(csv_path, index=False)
        print(
            "No Excel writer engine found. "
            f"Saved CSV fallback to {csv_path}. "
            "Install openpyxl or xlsxwriter if you need .xlsx output."
        )
