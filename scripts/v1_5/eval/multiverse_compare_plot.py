import argparse
import json
from pathlib import Path


def load_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_method(summary_path, label):
    summary = load_summary(summary_path)
    total = int(summary.get("num_reviews", 0))
    successful = int(summary.get("num_successful_reviews", 0))
    failed = int(summary.get("num_failed_reviews", 0))
    pass_rate_successful = float(summary.get("pass_rate", 0.0))
    score_coverage = (successful / total) if total else 0.0

    return {
        "label": label,
        "summary_path": str(summary_path),
        "total": total,
        "successful": successful,
        "failed": failed,
        "pass_rate_successful": pass_rate_successful,
        "score_coverage": score_coverage,
        "overall_score_mean": summary.get("means", {}).get("overall_score"),
    }


def svg_text(x, y, text, size=14, weight="normal", anchor="start", fill="#111111"):
    safe = (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" font-family="Arial, sans-serif" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{safe}</text>'
    )


def plot_comparison(methods, output_path):
    width, height = 1300, 700
    margin = 70
    chart_top = 120
    chart_bottom = 560
    chart_height = chart_bottom - chart_top
    panel_gap = 80
    panel_width = (width - 2 * margin - panel_gap) // 2

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(margin, 40, "MultiVerse: Baseline vs Retrieval from GPT-Scored Reviews", size=24, weight="bold"),
    ]

    def draw_panel(x0, title_text, value_key, colors, footer_kind):
        x1 = x0 + panel_width
        parts.append(f'<rect x="{x0}" y="{chart_top}" width="{panel_width}" height="{chart_height}" fill="none" stroke="#cccccc"/>')
        parts.append(svg_text(x0, chart_top - 18, title_text, size=18, weight="bold"))

        for tick in range(0, 101, 20):
            y = chart_bottom - int((tick / 100.0) * chart_height)
            parts.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="#e6e6e6" stroke-width="1"/>')
            parts.append(svg_text(x0 - 10, y + 5, tick, size=12, anchor="end", fill="#666666"))

        bar_width = 120
        centers = [x0 + panel_width * 0.32, x0 + panel_width * 0.72]

        for center, method, color in zip(centers, methods, colors):
            value = 100.0 * method[value_key]
            bar_height = int((value / 100.0) * chart_height)
            y0 = chart_bottom - bar_height
            x_left = int(center - bar_width / 2)
            parts.append(
                f'<rect x="{x_left}" y="{y0}" width="{bar_width}" height="{bar_height}" '
                f'fill="{color}" stroke="#111111"/>'
            )
            parts.append(svg_text(center, y0 - 8, f"{value:.1f}%", size=14, anchor="middle"))
            parts.append(svg_text(center, chart_bottom + 24, method["label"], size=14, weight="bold", anchor="middle"))

            if footer_kind == "scored":
                footer = f'{method["successful"]}/{method["total"]} scored'
            else:
                overall = method["overall_score_mean"]
                footer = f"mean score {'n/a' if overall is None else f'{overall:.2f}'}"
            parts.append(svg_text(center, chart_bottom + 44, footer, size=12, anchor="middle", fill="#444444"))

    draw_panel(
        margin,
        "GPT Pass Rate",
        "pass_rate_successful",
        ["#1f77b4", "#d62728"],
        "scored",
    )
    draw_panel(
        margin + panel_width + panel_gap,
        "GPT Scoring Coverage",
        "score_coverage",
        ["#6baed6", "#fb6a4a"],
        "mean",
    )

    note = (
        "Pass rate is measured over successfully scored items only. "
        "Coverage shows how many answers received a valid GPT evaluation."
    )
    parts.append(svg_text(margin, height - 30, note, size=13, fill="#666666"))
    parts.append("</svg>")

    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-summary",
        type=str,
        default="playground/data/eval/multiverse/summaries/multiverse_baseline_gpt-4.1-mini.json",
    )
    parser.add_argument(
        "--retrieval-summary",
        type=str,
        default="playground/data/eval/multiverse_retrieval/summaries/multiverse_retrieval_gpt-4.1-mini.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="playground/data/eval/multiverse/multiverse_baseline_vs_retrieval.svg",
    )
    args = parser.parse_args()

    methods = [
        summarize_method(Path(args.baseline_summary), "Baseline"),
        summarize_method(Path(args.retrieval_summary), "Retrieval"),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(methods, output_path)

    print(f"Saved plot to {output_path}")
    for method in methods:
        print(
            f"{method['label']}: pass_rate_successful={method['pass_rate_successful']:.4f}, "
            f"score_coverage={method['score_coverage']:.4f}, "
            f"successful={method['successful']}, total={method['total']}"
        )


if __name__ == "__main__":
    main()
