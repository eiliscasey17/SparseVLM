import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def svg_bar_chart(rows, output_path: Path, vision_tokens: int):
    text_tokens = [row["text_query_tokens"] for row in rows]
    prompt_tokens = [row["prompt_tokens"] for row in rows]

    avg_text = sum(text_tokens) / len(text_tokens)
    avg_prompt = sum(prompt_tokens) / len(prompt_tokens)
    pct_below_vision = 100.0 * sum(t < vision_tokens for t in text_tokens) / len(text_tokens)

    width = 920
    height = 560
    left = 110
    right = 40
    top = 70
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    labels = ["Avg text query", "Avg full prompt", "Vision tokens"]
    values = [avg_text, avg_prompt, float(vision_tokens)]
    colors = ["#4c78a8", "#72b7b2", "#e45756"]
    max_value = max(values) * 1.15

    bar_width = 140
    gap = (plot_w - bar_width * len(values)) / (len(values) + 1)

    def py(v):
        return top + plot_h - (v / max_value) * plot_h

    def bar_x(i):
        return left + gap + i * (bar_width + gap)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">MMBench: Text Tokens vs Vision Tokens</text>')
    parts.append(f'<text x="{width/2}" y="54" text-anchor="middle" font-size="13" font-family="Arial" fill="#555">Average text query length is far smaller than the standard 577 vision tokens</text>')

    for i in range(6):
        val = max_value * i / 5
        y = py(val)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{val:.0f}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')

    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = bar_x(i)
        y = py(value)
        h = top + plot_h - y
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" height="{h:.2f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width/2:.2f}" y="{y-10:.2f}" text-anchor="middle" font-size="16" font-family="Arial">{value:.1f}</text>')
        parts.append(f'<text x="{x + bar_width/2:.2f}" y="{top+plot_h+26}" text-anchor="middle" font-size="13" font-family="Arial">{label}</text>')

    note_x = left + plot_w * 0.55
    note_y = top + 40
    parts.append(f'<rect x="{note_x:.2f}" y="{note_y:.2f}" width="265" height="88" rx="10" fill="#f7f7f7" stroke="#dddddd"/>')
    parts.append(f'<text x="{note_x+14:.2f}" y="{note_y+24:.2f}" font-size="13" font-family="Arial">Samples: {len(rows)}</text>')
    parts.append(f'<text x="{note_x+14:.2f}" y="{note_y+46:.2f}" font-size="13" font-family="Arial">Avg text query tokens: {avg_text:.1f}</text>')
    parts.append(f'<text x="{note_x+14:.2f}" y="{note_y+68:.2f}" font-size="13" font-family="Arial">Text query &lt; 577 in {pct_below_vision:.1f}% of samples</text>')

    parts.append(f'<text x="{width/2}" y="{height-22}" text-anchor="middle" font-size="14" font-family="Arial">Token count</text>')
    parts.append('</svg>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("playground/data/eval/mmbench/metrics/mmbench_dev_20230712/llava-v1.5-13b-profile.jsonl"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("playground/data/eval/mmbench/mmbench_text_vs_vision_tokens.svg"),
    )
    parser.add_argument(
        "--vision-tokens",
        type=int,
        default=577,
    )
    args = parser.parse_args()

    rows = load_rows(args.metrics_file)
    svg_bar_chart(rows, args.output_file, args.vision_tokens)
    print(args.output_file)


if __name__ == "__main__":
    main()
