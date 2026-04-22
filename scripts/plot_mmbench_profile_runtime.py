import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f]


VISION_TOKEN_OFFSET = 577


def bucket_means(rows):
    buckets = [(0, 90), (90, 120), (120, 180), (180, 10**9)]
    points = []
    for lo, hi in buckets:
        chunk = [row for row in rows if lo <= row["prompt_tokens"] < hi]
        if not chunk:
            continue
        x = sum(row["prompt_tokens"] + VISION_TOKEN_OFFSET for row in chunk) / len(chunk)
        y = sum(row["elapsed_seconds"] for row in chunk) / len(chunk)
        label_hi = hi + VISION_TOKEN_OFFSET if hi < 10**9 else "max"
        label = f"{lo + VISION_TOKEN_OFFSET}-{label_hi}"
        points.append((x, y, label))
    return points


def scale(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return (out_min + out_max) / 2
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def svg_plot(rows, output_path: Path):
    width = 900
    height = 560
    left = 80
    right = 30
    top = 50
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [row["prompt_tokens"] + VISION_TOKEN_OFFSET for row in rows]
    ys = [row["elapsed_seconds"] for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    y_pad = max((y_max - y_min) * 0.08, 0.01)
    y_min = max(0.0, y_min - y_pad)
    y_max = y_max + y_pad

    bucket_points = bucket_means(rows)

    def px(x):
        return scale(x, x_min, x_max, left, left + plot_w)

    def py(y):
        return scale(y, y_min, y_max, top + plot_h, top)

    x_ticks = [660, 700, 760, 840, 940, 1080, 1240]
    y_ticks = 5

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">MMBench Baseline Profile: Total Context Tokens vs Runtime</text>')

    for tick in x_ticks:
        if tick < x_min or tick > x_max:
            continue
        x = px(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+22}" text-anchor="middle" font-size="12" font-family="Arial">{tick}</text>')

    for i in range(y_ticks + 1):
        val = y_min + (y_max - y_min) * i / y_ticks
        y = py(val)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{val:.3f}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')

    for row in rows:
            x = px(row["prompt_tokens"] + VISION_TOKEN_OFFSET)
            y = py(row["elapsed_seconds"])
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#1f77b4" fill-opacity="0.18"/>')

    if bucket_points:
        path_cmds = []
        for i, (xv, yv, _) in enumerate(bucket_points):
            cmd = "M" if i == 0 else "L"
            path_cmds.append(f"{cmd} {px(xv):.2f} {py(yv):.2f}")
        parts.append(f'<path d="{" ".join(path_cmds)}" fill="none" stroke="#d62728" stroke-width="2.5"/>')
        for xv, yv, label in bucket_points:
            x = px(xv)
            y = py(yv)
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" fill="#d62728"/>')
            parts.append(f'<text x="{x+6:.2f}" y="{y-8:.2f}" font-size="11" font-family="Arial" fill="#b22222">{label}</text>')

    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14" font-family="Arial">Text tokens + 577 vision tokens</text>')
    parts.append(f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {height/2})">Elapsed seconds</text>')
    parts.append('</svg>')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts))


def svg_bucket_line_plot(rows, output_path: Path):
    width = 900
    height = 560
    left = 80
    right = 30
    top = 50
    bottom = 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    points = bucket_means(rows)
    xs = [x for x, _, _ in points]
    ys = [y for _, y, _ in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_pad = max((x_max - x_min) * 0.1, 5)
    y_pad = max((y_max - y_min) * 0.25, 0.003)
    x_min -= x_pad
    x_max += x_pad
    y_min = max(0.0, y_min - y_pad)
    y_max += y_pad

    def px(x):
        return scale(x, x_min, x_max, left, left + plot_w)

    def py(y):
        return scale(y, y_min, y_max, top + plot_h, top)

    x_ticks = sorted({round(x) for x in xs})
    y_ticks = 5

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">MMBench Baseline Profile: Bucket Mean Runtime vs Total Context</text>')

    for tick in x_ticks:
        x = px(tick)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{top+plot_h+22}" text-anchor="middle" font-size="12" font-family="Arial">{tick}</text>')

    for i in range(y_ticks + 1):
        val = y_min + (y_max - y_min) * i / y_ticks
        y = py(val)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial">{val:.3f}</text>')

    parts.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>')

    path_cmds = []
    for i, (xv, yv, _) in enumerate(points):
        cmd = "M" if i == 0 else "L"
        path_cmds.append(f"{cmd} {px(xv):.2f} {py(yv):.2f}")
    parts.append(f'<path d="{" ".join(path_cmds)}" fill="none" stroke="#d62728" stroke-width="3"/>')

    for xv, yv, label in points:
        x = px(xv)
        y = py(yv)
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5.5" fill="#d62728"/>')
        parts.append(f'<text x="{x+8:.2f}" y="{y-10:.2f}" font-size="12" font-family="Arial" fill="#b22222">{label}</text>')

    parts.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14" font-family="Arial">Average text tokens + 577 vision tokens per bucket</text>')
    parts.append(f'<text x="20" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 20 {height/2})">Average elapsed seconds</text>')
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
        default=Path("playground/data/eval/mmbench/mmbench_profile_runtime_vs_context.svg"),
    )
    parser.add_argument(
        "--line-only",
        action="store_true",
        help="Render only the bucket-mean line, without the scatter points.",
    )
    args = parser.parse_args()

    rows = load_rows(args.metrics_file)
    if args.line_only:
        svg_bucket_line_plot(rows, args.output_file)
    else:
        svg_plot(rows, args.output_file)
    print(args.output_file)


if __name__ == "__main__":
    main()
