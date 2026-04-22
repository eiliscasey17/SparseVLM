import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]


def run_command(cmd, env=None):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


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


def summarize_retrieval_answers(answers_file):
    selected = []
    kept_pct = []
    sink_tokens = []
    retrieved_tokens = []

    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            metadata = item.get("metadata", {})
            if "selected_vision_tokens" in metadata:
                selected.append(float(metadata["selected_vision_tokens"]))
            if "kept_vision_token_pct" in metadata:
                kept_pct.append(float(metadata["kept_vision_token_pct"]))
            if "sink_tokens_kept" in metadata:
                sink_tokens.append(float(metadata["sink_tokens_kept"]))
            if "retrieved_non_sink_tokens" in metadata:
                retrieved_tokens.append(float(metadata["retrieved_non_sink_tokens"]))

    def mean_or_zero(values):
        return sum(values) / len(values) if values else 0.0

    return {
        "avg_selected_vision_tokens": mean_or_zero(selected),
        "avg_kept_vision_token_pct": mean_or_zero(kept_pct),
        "avg_sink_tokens_kept": mean_or_zero(sink_tokens),
        "avg_retrieved_non_sink_tokens": mean_or_zero(retrieved_tokens),
    }


def build_sink_schedule(total_budget, step):
    schedule = list(range(total_budget, -1, -step))
    if schedule[-1] != 0:
        schedule.append(0)
    return schedule


def run_baseline(args, answers_dir):
    experiment = f"mmbench_baseline_ag{int(args.baseline_agnostic_sparsity * 100)}_aw{int(args.baseline_aware_sparsity * 100)}"
    answers_file = answers_dir / f"{experiment}.jsonl"

    cmd = [
        sys.executable,
        "-m",
        "llava.eval.model_vqa_mmbench",
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
        "--agnostic-sparsity",
        str(args.baseline_agnostic_sparsity),
        "--aware-sparsity",
        str(args.baseline_aware_sparsity),
    ]

    if args.model_base:
        cmd.extend(["--model-base", args.model_base])

    run_command(cmd)
    metrics = compute_accuracy(answers_file, args.question_file, experiment)
    return {
        "method": "agnostic_plus_aware",
        "experiment": experiment,
        "answers_file": str(answers_file),
        "agnostic_sparsity": args.baseline_agnostic_sparsity,
        "aware_sparsity": args.baseline_aware_sparsity,
        "configured_token_budget": args.total_token_budget,
        "sink_tokens_configured": None,
        "retrieval_tokens_configured": None,
        **metrics,
    }


def run_retrieval_sweep(args, answers_dir):
    results = []
    for sink_tokens in build_sink_schedule(args.total_token_budget, args.sink_step):
        retrieval_tokens = max(0, args.total_token_budget - sink_tokens)
        experiment = (
            f"mmbench_retrieval_pr{int(args.retrieval_prune_ratio * 100)}"
            f"_budget{args.total_token_budget}_sink{sink_tokens}_ret{retrieval_tokens}"
        )
        answers_file = answers_dir / f"{experiment}.jsonl"

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
            str(args.retrieval_prune_ratio),
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
            str(args.total_token_budget),
        ]

        if args.model_base:
            cmd.extend(["--model-base", args.model_base])
        if args.cluster_page_size is not None:
            cmd.extend(["--cluster-page-size", str(args.cluster_page_size)])
        if args.use_projected_tokens_for_output:
            cmd.append("--use-projected-tokens-for-output")

        run_command(cmd)
        metrics = compute_accuracy(answers_file, args.question_file, experiment)
        token_stats = summarize_retrieval_answers(answers_file)

        results.append({
            "method": "sink_plus_retrieval",
            "experiment": experiment,
            "answers_file": str(answers_file),
            "retrieval_prune_ratio": args.retrieval_prune_ratio,
            "configured_token_budget": args.total_token_budget,
            "sink_tokens_configured": sink_tokens,
            "retrieval_tokens_configured": retrieval_tokens,
            **token_stats,
            **metrics,
        })

    return results


def plot_results(args, baseline_result, retrieval_results, plot_file):
    retrieval_results = sorted(
        retrieval_results,
        key=lambda item: item["avg_retrieved_non_sink_tokens"],
    )

    x_vals = [item["avg_retrieved_non_sink_tokens"] for item in retrieval_results]
    y_vals = [item["accuracy"] for item in retrieval_results]

    plt.figure(figsize=(10, 6))
    plt.plot(
        x_vals,
        y_vals,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Sink + retrieval sweep",
    )

    for item in retrieval_results:
        plt.annotate(
            f"S={item['sink_tokens_configured']}",
            (item["avg_retrieved_non_sink_tokens"], item["accuracy"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )

    plt.axhline(
        baseline_result["accuracy"],
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=(
            "Baseline agnostic+aware "
            f"({baseline_result['agnostic_sparsity']:.2f} + {baseline_result['aware_sparsity']:.2f})"
        ),
    )

    plt.title(
        "MMBench Pareto Curve at Matched Final Vision-Token Budget "
        f"(budget={args.total_token_budget})"
    )
    plt.xlabel("Average retrieved non-sink vision tokens")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_file, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--split", type=str, default="mmbench_dev_20230712")
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--output-dir", type=str, default="playground/data/eval/mmbench/pareto_curve")

    parser.add_argument("--vision-token-count", type=int, default=576)
    parser.add_argument("--keep-ratio", type=float, default=None)
    parser.add_argument("--total-token-budget", type=int, default=None)

    parser.add_argument("--baseline-agnostic-sparsity", type=float, default=0.5)
    parser.add_argument("--baseline-aware-sparsity", type=float, default=0.5)

    parser.add_argument("--retrieval-prune-ratio", type=float, default=0.5)
    parser.add_argument("--sink-step", type=int, default=32)
    parser.add_argument("--num-clusters", type=int, default=16)
    parser.add_argument("--clustering-mode", type=str, default="semantic", choices=["kmeans", "semantic"])
    parser.add_argument("--cluster-page-size", type=int, default=4096)
    parser.add_argument("--topk-clusters", type=int, default=4)
    parser.add_argument("--retrieval-mode", type=str, default="cosine_mean")
    parser.add_argument("--use-projected-tokens-for-output", action="store_true")
    args = parser.parse_args()

    if args.question_file is None:
        args.question_file = f"./playground/data/eval/mmbench/{args.split}.tsv"

    if args.total_token_budget is not None:
        args.total_token_budget = int(args.total_token_budget)
        args.keep_ratio = args.total_token_budget / float(args.vision_token_count)
    elif args.keep_ratio is not None:
        args.total_token_budget = int(round(args.vision_token_count * args.keep_ratio))
    else:
        effective_keep_ratio = (
            (1.0 - args.baseline_agnostic_sparsity)
            * (1.0 - args.baseline_aware_sparsity)
        )
        args.keep_ratio = effective_keep_ratio
        args.total_token_budget = int(round(args.vision_token_count * effective_keep_ratio))

    output_dir = Path(args.output_dir)
    answers_dir = output_dir / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)

    baseline_result = run_baseline(args, answers_dir)
    retrieval_results = run_retrieval_sweep(args, answers_dir)

    results = {
        "config": {
            "model_path": args.model_path,
            "model_base": args.model_base,
            "split": args.split,
            "question_file": args.question_file,
            "conv_mode": args.conv_mode,
            "vision_token_count": args.vision_token_count,
            "keep_ratio": args.keep_ratio,
            "total_token_budget": args.total_token_budget,
            "baseline_agnostic_sparsity": args.baseline_agnostic_sparsity,
            "baseline_aware_sparsity": args.baseline_aware_sparsity,
            "retrieval_prune_ratio": args.retrieval_prune_ratio,
            "sink_step": args.sink_step,
            "num_clusters": args.num_clusters,
            "clustering_mode": args.clustering_mode,
            "cluster_page_size": args.cluster_page_size,
            "topk_clusters": args.topk_clusters,
            "retrieval_mode": args.retrieval_mode,
            "use_projected_tokens_for_output": args.use_projected_tokens_for_output,
        },
        "baseline": baseline_result,
        "retrieval_sweep": retrieval_results,
    }

    results_file = output_dir / "mmbench_pareto_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_file = output_dir / "mmbench_pareto_curve.png"
    plot_results(args, baseline_result, retrieval_results, plot_file)

    print(f"Saved results to {results_file}")
    print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    main()
