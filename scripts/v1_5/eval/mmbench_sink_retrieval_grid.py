import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def parse_percentage_schedule(raw_value):
    if raw_value is None:
        return list(range(0, 101, 10))

    values = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value < 0 or value > 100:
            raise ValueError(f"Percentages must be in [0, 100], got {value}.")
        values.append(value)

    if not values:
        raise ValueError("Percentage schedule cannot be empty.")

    return sorted(set(values))


def pct_to_count(total_tokens, percentage):
    return int(round(total_tokens * (percentage / 100.0)))


def run_command(cmd, env=None):
    print("Running:", " ".join(cmd))
    start_time = time.perf_counter()
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    return time.perf_counter() - start_time


def compute_accuracy(pred_file, gt_file, experiment_name):
    cmd = [
        sys.executable,
        "playground/data/eval/mmbench/compute_accuracy.py",
        "--pred-file",
        str(pred_file),
        "--gt-file",
        str(gt_file),
        "--experiment-name",
        experiment_name,
    ]
    output = subprocess.check_output(cmd, cwd=REPO_ROOT, text=True).strip()
    correct_str, total_str, accuracy_str, _ = [part.strip() for part in output.split("|", 3)]
    return {
        "correct": int(correct_str),
        "total": int(total_str),
        "accuracy": float(accuracy_str),
        "raw_output": output,
    }


def summarize_answers(answers_file):
    aggregate = defaultdict(list)

    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            metadata = item.get("metadata", {})
            for key in (
                "total_vision_tokens",
                "kept_after_prune_tokens",
                "available_non_sink_tokens",
                "selected_vision_tokens",
                "sink_tokens_kept",
                "retrieved_non_sink_tokens",
                "kept_vision_token_pct",
            ):
                if key in metadata:
                    aggregate[key].append(float(metadata[key]))

    summary = {}
    for key, values in aggregate.items():
        summary[f"avg_{key}"] = sum(values) / len(values) if values else 0.0

    return summary


def build_experiment_name(prune_ratio_pct, sink_pct, retrieval_pct, sink_tokens, retrieval_tokens):
    return (
        f"mmbench_grid_pr{prune_ratio_pct:03d}"
        f"_sinkpct{sink_pct:03d}"
        f"_retpct{retrieval_pct:03d}"
        f"_sinktok{sink_tokens:03d}"
        f"_rettok{retrieval_tokens:03d}"
    )


def load_experiment_metrics(metrics_file):
    if not metrics_file.exists():
        return {}

    with open(metrics_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_experiment_metrics(metrics_file, payload):
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_single_experiment(args, output_dir, sink_pct, retrieval_pct):
    nominal_kept_after_prune = pct_to_count(args.vision_token_count, 100 - args.prune_ratio_pct)
    sink_tokens = pct_to_count(nominal_kept_after_prune, sink_pct)
    nominal_non_sink_tokens = max(0, nominal_kept_after_prune - sink_tokens)
    retrieval_tokens = pct_to_count(nominal_non_sink_tokens, retrieval_pct)
    token_budget = sink_tokens + retrieval_tokens

    experiment = build_experiment_name(
        prune_ratio_pct=args.prune_ratio_pct,
        sink_pct=sink_pct,
        retrieval_pct=retrieval_pct,
        sink_tokens=sink_tokens,
        retrieval_tokens=retrieval_tokens,
    )
    answers_file = output_dir / "answers" / f"{experiment}.jsonl"
    metrics_file = output_dir / "metrics" / f"{experiment}.json"
    command_elapsed_seconds = None

    if not args.plot_only and (args.rerun_existing or not answers_file.exists()):
        cmd = [
            sys.executable,
            "-m",
            "llava.eval.model_vqa_mmbench_retrieval",
            "--model-path",
            args.model_path,
            "--question-file",
            args.question_file,
            "--answers-file",
            str(answers_file),
            "--single-pred-prompt",
            "--temperature",
            "0",
            "--conv-mode",
            args.conv_mode,
            "--prune-ratio",
            str(args.prune_ratio),
            "--num-sink-tokens",
            str(sink_tokens),
            "--num-clusters",
            str(args.num_clusters),
            "--clustering-mode",
            args.clustering_mode,
            "--topk-clusters",
            str(args.topk_clusters),
            "--retrieval-mode",
            args.retrieval_mode,
            "--token-budget",
            str(token_budget),
        ]

        if args.model_base:
            cmd.extend(["--model-base", args.model_base])
        if args.cluster_page_size is not None:
            cmd.extend(["--cluster-page-size", str(args.cluster_page_size)])
        if args.use_projected_tokens_for_output:
            cmd.append("--use-projected-tokens-for-output")

        command_elapsed_seconds = run_command(cmd)
        save_experiment_metrics(
            metrics_file,
            {
                "experiment": experiment,
                "command_elapsed_seconds": command_elapsed_seconds,
                "sink_pct_of_pruned_remaining": sink_pct,
                "retrieval_pct_of_non_sink_remaining": retrieval_pct,
                "configured_sink_tokens": sink_tokens,
                "configured_retrieval_tokens": retrieval_tokens,
                "configured_token_budget": token_budget,
            },
        )
    elif not answers_file.exists():
        raise FileNotFoundError(
            f"Missing answers file for plot-only mode: {answers_file}"
        )
    else:
        print(f"Reusing existing answers: {answers_file}")
        metrics_payload = load_experiment_metrics(metrics_file)
        command_elapsed_seconds = metrics_payload.get("command_elapsed_seconds")

    accuracy = compute_accuracy(
        pred_file=answers_file,
        gt_file=args.question_file,
        experiment_name=experiment,
    )
    summary = summarize_answers(answers_file)

    record = {
        "experiment": experiment,
        "answers_file": str(answers_file),
        "prune_ratio": args.prune_ratio,
        "prune_ratio_pct": args.prune_ratio_pct,
        "vision_token_count": args.vision_token_count,
        "nominal_kept_after_prune_tokens": nominal_kept_after_prune,
        "sink_pct_of_pruned_remaining": sink_pct,
        "retrieval_pct_of_non_sink_remaining": retrieval_pct,
        "configured_sink_tokens": sink_tokens,
        "configured_retrieval_tokens": retrieval_tokens,
        "configured_token_budget": token_budget,
        "num_clusters": args.num_clusters,
        "clustering_mode": args.clustering_mode,
        "cluster_page_size": args.cluster_page_size,
        "topk_clusters": args.topk_clusters,
        "retrieval_mode": args.retrieval_mode,
        "use_projected_tokens_for_output": args.use_projected_tokens_for_output,
        "command_elapsed_seconds": command_elapsed_seconds,
        **summary,
        **accuracy,
    }
    return record


def write_results(records, output_dir):
    json_file = output_dir / "mmbench_sink_retrieval_grid_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    tsv_file = output_dir / "mmbench_sink_retrieval_grid_results.tsv"
    fieldnames = [
        "experiment",
        "answers_file",
        "prune_ratio",
        "prune_ratio_pct",
        "vision_token_count",
        "nominal_kept_after_prune_tokens",
        "sink_pct_of_pruned_remaining",
        "retrieval_pct_of_non_sink_remaining",
        "configured_sink_tokens",
        "configured_retrieval_tokens",
        "configured_token_budget",
        "num_clusters",
        "clustering_mode",
        "cluster_page_size",
        "topk_clusters",
        "retrieval_mode",
        "use_projected_tokens_for_output",
        "command_elapsed_seconds",
        "avg_total_vision_tokens",
        "avg_kept_after_prune_tokens",
        "avg_available_non_sink_tokens",
        "avg_selected_vision_tokens",
        "avg_sink_tokens_kept",
        "avg_retrieved_non_sink_tokens",
        "avg_kept_vision_token_pct",
        "correct",
        "total",
        "accuracy",
        "raw_output",
    ]
    with open(tsv_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)

    return json_file, tsv_file


def _build_metric_grid(records, metric_key):
    sink_values = sorted({record["sink_pct_of_pruned_remaining"] for record in records})
    retrieval_values = sorted({record["retrieval_pct_of_non_sink_remaining"] for record in records})
    grid = []

    for sink_pct in sink_values:
        row = []
        for retrieval_pct in retrieval_values:
            match = next(
                (
                    record for record in records
                    if record["sink_pct_of_pruned_remaining"] == sink_pct
                    and record["retrieval_pct_of_non_sink_remaining"] == retrieval_pct
                ),
                None,
            )
            row.append(None if match is None else match.get(metric_key))
        grid.append(row)

    return sink_values, retrieval_values, grid


def plot_metric_heatmap(records, output_dir, metric_key, title, colorbar_label, output_stem):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib and numpy are required to generate the sink/retrieval sweep plots. "
            "Please install them in the active environment, or rerun with --plot-only after installing."
        ) from exc

    sink_values, retrieval_values, grid = _build_metric_grid(records, metric_key)
    matrix = np.array(
        [
            [np.nan if value is None else float(value) for value in row]
            for row in grid
        ],
        dtype=float,
    )

    finite_values = matrix[np.isfinite(matrix)]
    text_switch_threshold = None
    if finite_values.size > 0:
        text_switch_threshold = float(finite_values.max()) * 0.45

    plt.figure(figsize=(10, 6.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9")
    image = plt.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    colorbar = plt.colorbar(image)
    colorbar.set_label(colorbar_label)

    plt.xticks(range(len(retrieval_values)), retrieval_values)
    plt.yticks(range(len(sink_values)), sink_values)
    plt.xlabel("Retrieval percentage of remaining non-sink tokens")
    plt.ylabel(f"Sink percentage of tokens kept after {records[0]['prune_ratio_pct']}% prune")
    plt.title(title)

    for row_idx, sink_pct in enumerate(sink_values):
        for col_idx, retrieval_pct in enumerate(retrieval_values):
            value = matrix[row_idx, col_idx]
            if np.isnan(value):
                label = "NA"
            elif metric_key == "accuracy":
                label = f"{value:.3f}"
            else:
                label = f"{value:.1f}s"
            plt.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color="white" if text_switch_threshold is not None and not np.isnan(value) and value >= text_switch_threshold else "black",
                fontsize=8,
            )

    plt.tight_layout()
    png_file = output_dir / f"{output_stem}.png"
    svg_file = output_dir / f"{output_stem}.svg"
    plt.savefig(png_file, dpi=220)
    plt.savefig(svg_file)
    plt.close()
    return png_file, svg_file


def plot_results(records, output_dir, use_actual_x=False):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the sink/retrieval sweep plots. "
            "Please install it in the active environment, or rerun with --plot-only after installing."
        ) from exc

    grouped = defaultdict(list)
    for record in records:
        grouped[record["sink_pct_of_pruned_remaining"]].append(record)

    plt.figure(figsize=(11, 7))
    cmap = plt.get_cmap("viridis")
    sink_values = sorted(grouped)
    denom = max(1, len(sink_values) - 1)

    for index, sink_pct in enumerate(sink_values):
        series = sorted(
            grouped[sink_pct],
            key=lambda item: item["avg_retrieved_non_sink_tokens"] if use_actual_x else item["configured_retrieval_tokens"],
        )
        x_values = [
            item["avg_retrieved_non_sink_tokens"] if use_actual_x else item["configured_retrieval_tokens"]
            for item in series
        ]
        y_values = [item["accuracy"] for item in series]
        color = cmap(index / denom)
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color,
            label=f"Sink {sink_pct}%",
        )

    plt.title(
        f"MMBench Accuracy vs Retrieval Tokens at Fixed {records[0]['prune_ratio_pct']}% Pruning"
    )
    plt.xlabel(
        "Average retrieved non-sink tokens" if use_actual_x else "Configured retrieval tokens"
    )
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Sink share", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    suffix = "actualx" if use_actual_x else "configuredx"
    png_file = output_dir / f"mmbench_sink_retrieval_grid_{suffix}.png"
    svg_file = output_dir / f"mmbench_sink_retrieval_grid_{suffix}.svg"
    plt.savefig(png_file, dpi=220)
    plt.savefig(svg_file)
    plt.close()
    return png_file, svg_file


def plot_vs_total_tokens(records, output_dir, metric_key, y_label, title, output_stem):
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate the sink/retrieval sweep plots. "
            "Please install it in the active environment, or rerun with --plot-only after installing."
        ) from exc

    grouped = defaultdict(list)
    for record in records:
        grouped[record["sink_pct_of_pruned_remaining"]].append(record)

    plt.figure(figsize=(11, 7))
    cmap = plt.get_cmap("viridis")
    sink_values = sorted(grouped)
    denom = max(1, len(sink_values) - 1)
    for index, sink_pct in enumerate(sink_values):
        series = sorted(
            grouped[sink_pct],
            key=lambda item: item["configured_token_budget"],
        )
        x_values = [item["configured_token_budget"] for item in series]
        y_values = [item[metric_key] for item in series]
        color = cmap(index / denom)
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2,
            markersize=5,
            color=color,
            label=f"Sink {sink_pct}%",
        )

    plt.title(title)
    plt.xlabel("Configured total visual tokens used (sink + retrieval)")
    plt.ylabel(y_label)
    plt.xlim(0, 300)
    plt.grid(True, alpha=0.25)
    plt.legend(title="Sink share", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    png_file = output_dir / f"{output_stem}.png"
    svg_file = output_dir / f"{output_stem}.svg"
    plt.savefig(png_file, dpi=220)
    plt.savefig(svg_file)
    plt.close()
    return png_file, svg_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--split", type=str, default="mmbench_dev_20230712")
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="playground/data/eval/mmbench/sink_retrieval_grid",
    )

    parser.add_argument("--vision-token-count", type=int, default=576)
    parser.add_argument("--prune-ratio", type=float, default=0.5)
    parser.add_argument("--sink-percentages", type=str, default=None)
    parser.add_argument("--retrieval-percentages", type=str, default=None)

    parser.add_argument("--num-clusters", type=int, default=16)
    parser.add_argument(
        "--clustering-mode",
        type=str,
        default="kmeans",
        choices=["kmeans", "semantic"],
    )
    parser.add_argument("--cluster-page-size", type=int, default=4096)
    parser.add_argument("--topk-clusters", type=int, default=4)
    parser.add_argument("--retrieval-mode", type=str, default="cosine_mean")
    parser.add_argument("--use-projected-tokens-for-output", action="store_true")

    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    if not math.isclose(args.prune_ratio * 100.0, round(args.prune_ratio * 100.0), abs_tol=1e-9):
        raise ValueError(
            "This sweep script expects prune_ratio to map cleanly to a percentage. "
            f"Received {args.prune_ratio}."
        )

    args.prune_ratio_pct = int(round(args.prune_ratio * 100.0))
    if args.question_file is None:
        args.question_file = f"./playground/data/eval/mmbench/{args.split}.tsv"

    sink_percentages = parse_percentage_schedule(args.sink_percentages)
    retrieval_percentages = parse_percentage_schedule(args.retrieval_percentages)

    output_dir = Path(args.output_dir)
    (output_dir / "answers").mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "split": args.split,
        "question_file": args.question_file,
        "conv_mode": args.conv_mode,
        "vision_token_count": args.vision_token_count,
        "prune_ratio": args.prune_ratio,
        "sink_percentages": sink_percentages,
        "retrieval_percentages": retrieval_percentages,
        "num_clusters": args.num_clusters,
        "clustering_mode": args.clustering_mode,
        "cluster_page_size": args.cluster_page_size,
        "topk_clusters": args.topk_clusters,
        "retrieval_mode": args.retrieval_mode,
        "use_projected_tokens_for_output": args.use_projected_tokens_for_output,
        "plot_only": args.plot_only,
        "rerun_existing": args.rerun_existing,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    records = []
    total_runs = len(sink_percentages) * len(retrieval_percentages)
    current_run = 0
    for sink_pct in sink_percentages:
        for retrieval_pct in retrieval_percentages:
            current_run += 1
            print(
                f"[{current_run}/{total_runs}] sink={sink_pct}% "
                f"retrieval={retrieval_pct}%"
            )
            record = run_single_experiment(
                args=args,
                output_dir=output_dir,
                sink_pct=sink_pct,
                retrieval_pct=retrieval_pct,
            )
            records.append(record)

    records = sorted(
        records,
        key=lambda item: (
            item["sink_pct_of_pruned_remaining"],
            item["retrieval_pct_of_non_sink_remaining"],
        ),
    )
    json_file, tsv_file = write_results(records, output_dir)
    runtime_plot_png, runtime_plot_svg = plot_metric_heatmap(
        records,
        output_dir,
        metric_key="command_elapsed_seconds",
        title="MMBench Total Runtime by Sink/Retrieval Setting",
        colorbar_label="Runtime over full dataset (seconds)",
        output_stem="mmbench_sink_retrieval_grid_runtime",
    )
    accuracy_plot_png, accuracy_plot_svg = plot_metric_heatmap(
        records,
        output_dir,
        metric_key="accuracy",
        title="MMBench Accuracy by Sink/Retrieval Setting",
        colorbar_label="Accuracy",
        output_stem="mmbench_sink_retrieval_grid_accuracy",
    )
    accuracy_tokens_plot_png, accuracy_tokens_plot_svg = plot_vs_total_tokens(
        records,
        output_dir,
        metric_key="accuracy",
        y_label="Accuracy",
        title="MMBench Accuracy vs Total Visual Tokens Used",
        output_stem="mmbench_sink_retrieval_grid_accuracy_vs_total_tokens",
    )
    runtime_tokens_plot_png, runtime_tokens_plot_svg = plot_vs_total_tokens(
        records,
        output_dir,
        metric_key="command_elapsed_seconds",
        y_label="Runtime over full dataset (seconds)",
        title="MMBench Runtime vs Total Visual Tokens Used",
        output_stem="mmbench_sink_retrieval_grid_runtime_vs_total_tokens",
    )
    configured_plot_png, configured_plot_svg = plot_results(
        records,
        output_dir,
        use_actual_x=False,
    )
    actual_plot_png, actual_plot_svg = plot_results(
        records,
        output_dir,
        use_actual_x=True,
    )

    print(f"Saved manifest to {output_dir / 'manifest.json'}")
    print(f"Saved results JSON to {json_file}")
    print(f"Saved results TSV to {tsv_file}")
    print(f"Saved runtime heatmap to {runtime_plot_png} and {runtime_plot_svg}")
    print(f"Saved accuracy heatmap to {accuracy_plot_png} and {accuracy_plot_svg}")
    print(
        f"Saved accuracy-vs-total-tokens plot to "
        f"{accuracy_tokens_plot_png} and {accuracy_tokens_plot_svg}"
    )
    print(
        f"Saved runtime-vs-total-tokens plot to "
        f"{runtime_tokens_plot_png} and {runtime_tokens_plot_svg}"
    )
    print(f"Saved configured-x plot to {configured_plot_png} and {configured_plot_svg}")
    print(f"Saved actual-x plot to {actual_plot_png} and {actual_plot_svg}")


if __name__ == "__main__":
    main()
