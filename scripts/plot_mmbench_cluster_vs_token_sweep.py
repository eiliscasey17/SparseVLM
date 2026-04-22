import argparse
import csv
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def load_cluster_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "method": row["method"],
                    "clusters_kept": int(row["clusters_kept"]),
                    "clusters_total": int(row["clusters_total"]),
                    "questions": int(row["questions"]),
                    "correct": int(row["correct"]),
                    "accuracy": float(row["accuracy"]),
                    "avg_kept_vision_tokens": float(row["avg_kept_vision_tokens"]),
                }
            )
    return rows


def load_token_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "method": "individual_token",
                    "label": row.get("retention_target", ""),
                    "accuracy": float(row["accuracy"]),
                    "avg_kept_vision_tokens": float(row["avg_kept_vision_tokens"]),
                }
            )
    return rows


def load_encoder_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "method": "encoder_token",
                    "label": row.get("retention_target", ""),
                    "accuracy": float(row["accuracy"]),
                    "avg_kept_vision_tokens": float(row["avg_kept_vision_tokens"]),
                }
            )
    return rows


def aggregate_cluster_rows(rows, min_questions=100):
    grouped = defaultdict(lambda: {"questions": 0, "correct": 0, "kept_tokens_sum": 0.0})
    for row in rows:
        if row["clusters_total"] <= 0:
            continue
        retained_cluster_ratio = row["clusters_kept"] / row["clusters_total"]
        key = (row["method"], round(retained_cluster_ratio, 4))
        grouped[key]["questions"] += row["questions"]
        grouped[key]["correct"] += row["correct"]
        grouped[key]["kept_tokens_sum"] += row["avg_kept_vision_tokens"] * row["questions"]

    aggregated = []
    for (method, retained_cluster_ratio), stats in grouped.items():
        if stats["questions"] < min_questions:
            continue
        aggregated.append(
            {
                "method": method,
                "retained_cluster_ratio": retained_cluster_ratio,
                "questions": stats["questions"],
                "accuracy": stats["correct"] / stats["questions"],
                "avg_kept_vision_tokens": stats["kept_tokens_sum"] / stats["questions"],
            }
        )
    return aggregated


def scale(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return (out_min + out_max) / 2
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def build_series(rows, method, label, color, marker):
    points = [row for row in rows if row["method"] == method]
    points = sorted(points, key=lambda row: row["avg_kept_vision_tokens"], reverse=True)
    return {
        "label": label,
        "color": color,
        "marker": marker,
        "points": points,
    }


def render_svg(series_list, output_path: Path, title: str):
    width = 980
    height = 620
    left = 80
    right = 40
    top = 70
    bottom = 80
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_points = [pt for series in series_list for pt in series["points"]]
    xs = [pt["avg_kept_vision_tokens"] for pt in all_points]
    ys = [pt["accuracy"] for pt in all_points]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    x_pad = max((x_max - x_min) * 0.05, 1.0)
    y_pad = max((y_max - y_min) * 0.10, 0.01)
    x_min -= x_pad
    x_max += x_pad
    y_min = max(0.0, y_min - y_pad)
    y_max = min(1.0, y_max + y_pad)

    def px(x):
        return scale(x, x_min, x_max, left, left + plot_w)

    def py(y):
        return scale(y, y_min, y_max, top + plot_h, top)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="34" text-anchor="middle" font-size="22" font-family="Arial">{title}</text>')

    for i in range(6):
        value = x_min + (x_max - x_min) * i / 5
        x = px(value)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#ececec" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+24}" text-anchor="middle" font-size="12" font-family="Arial">{value:.0f}</text>')

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = py(value)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#ececec" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{value:.3f}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')

    for series in series_list:
        if not series["points"]:
            continue
        path_cmds = []
        for idx, pt in enumerate(series["points"]):
            cmd = "M" if idx == 0 else "L"
            path_cmds.append(f"{cmd} {px(pt['avg_kept_vision_tokens']):.2f} {py(pt['accuracy']):.2f}")
        parts.append(f'<path d="{" ".join(path_cmds)}" fill="none" stroke="{series["color"]}" stroke-width="2.5"/>')
        for pt in series["points"]:
            x = px(pt["avg_kept_vision_tokens"])
            y = py(pt["accuracy"])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="{series["color"]}"/>')

    legend_x = left + 10
    legend_y = top + 14
    for idx, series in enumerate(series_list):
        y = legend_y + idx * 22
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{series["color"]}" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{legend_x+14}" cy="{y}" r="4.5" fill="{series["color"]}"/>')
        parts.append(f'<text x="{legend_x+38}" y="{y+4}" font-size="12" font-family="Arial">{series["label"]}</text>')

    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14" font-family="Arial">Average kept vision tokens</text>')
    parts.append(f'<text x="22" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 22 {height/2})">Accuracy</text>')
    parts.append('</svg>')
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render(output_path: Path, title: str, series_list):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".svg":
        render_svg(series_list, output_path, title)
        return

    if output_path.suffix.lower() == ".png":
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_svg = Path(tmpdir) / "plot.svg"
            render_svg(series_list, tmp_svg, title)
            subprocess.run(
                ["rsvg-convert", "-o", str(output_path), str(tmp_svg)],
                check=True,
            )
        return

    raise ValueError(f"Unsupported output format: {output_path.suffix}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-summary-file", type=Path, required=True)
    parser.add_argument("--token-summary-file", type=Path, required=True)
    parser.add_argument("--encoder-summary-file", type=Path, default=None)
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("playground/data/eval/mmbench/cluster_sweep/cluster_vs_token_accuracy_aggregated.png"),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="MMBench Accuracy vs Kept Vision Tokens (Aggregated Cluster Curves)",
    )
    parser.add_argument("--min-cluster-questions", type=int, default=100)
    args = parser.parse_args()

    cluster_rows = aggregate_cluster_rows(
        load_cluster_rows(args.cluster_summary_file),
        min_questions=args.min_cluster_questions,
    )
    token_rows = load_token_rows(args.token_summary_file)
    encoder_rows = load_encoder_rows(args.encoder_summary_file) if args.encoder_summary_file else []

    series_list = [
        build_series(cluster_rows, "kmeans_cluster_llm", "K-means clusters scored by LLM", "#1f77b4", "o"),
        build_series(cluster_rows, "semantic_page_cluster_llm", "Semantic clusters scored by LLM", "#d62728", "s"),
        build_series(cluster_rows, "kmeans_cluster_encoder", "K-means clusters scored by encoder", "#17becf", "D"),
        build_series(cluster_rows, "semantic_page_cluster_encoder", "Semantic clusters scored by encoder", "#ff7f0e", "v"),
        build_series(token_rows, "individual_token", "Individual token removal", "#2ca02c", "^"),
        build_series(encoder_rows, "encoder_token", "Encoder token removal", "#9467bd", "D"),
    ]
    render(args.output_file, args.title, series_list)
    print(args.output_file)


if __name__ == "__main__":
    main()
