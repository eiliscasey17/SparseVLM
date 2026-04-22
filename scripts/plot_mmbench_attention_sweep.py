import argparse
import csv
from pathlib import Path


def load_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "retention_target": row.get("retention_target", ""),
                    "target_type": row.get("target_type", "ratio"),
                    "target_value": float(row.get("target_value", row["retained_ratio"])),
                    "retained_ratio": float(row["retained_ratio"]),
                    "questions": int(row["questions"]),
                    "correct": int(row["correct"]),
                    "accuracy": float(row["accuracy"]),
                    "avg_kept_vision_tokens": float(row["avg_kept_vision_tokens"]),
                    "avg_full_vision_tokens": float(row["avg_full_vision_tokens"]),
                    "total_generation_seconds": float(row["total_generation_seconds"]),
                    "avg_generation_seconds": float(row["avg_generation_seconds"]),
                    "avg_generated_tokens": float(row["avg_generated_tokens"]),
                }
            )
    return rows


def scale(value, in_min, in_max, out_min, out_max):
    if in_max == in_min:
        return (out_min + out_max) / 2
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def format_ratio_label(ratio):
    return f"{ratio * 100:.0f}%"


def format_x_tick_label(row):
    if row["target_type"] == "count":
        count = int(round(row["target_value"]))
        return f"{count} tok"
    return f"{row['target_value'] * 100:.0f}%"


def svg_plot(rows, output_path: Path, title: str):
    rows = sorted(rows, key=lambda row: row["retained_ratio"], reverse=True)

    width = 980
    height = 580
    left = 90
    right = 90
    top = 60
    bottom = 90
    plot_w = width - left - right
    plot_h = height - top - bottom

    xs = [row["retained_ratio"] * 100.0 for row in rows]
    ys_accuracy = [row["accuracy"] for row in rows]
    ys_time = [row["avg_generation_seconds"] for row in rows]

    x_min = 0.0
    x_max = 100.0

    acc_min = min(ys_accuracy)
    acc_max = max(ys_accuracy)
    acc_pad = max((acc_max - acc_min) * 0.12, 0.01)
    acc_min = max(0.0, acc_min - acc_pad)
    acc_max = min(1.0, acc_max + acc_pad)

    time_min = min(ys_time)
    time_max = max(ys_time)
    time_pad = max((time_max - time_min) * 0.12, 0.01)
    time_min = max(0.0, time_min - time_pad)
    time_max += time_pad

    def px(x):
        return scale(x, x_max, x_min, left, left + plot_w)

    def py_acc(y):
        return scale(y, acc_min, acc_max, top + plot_h, top)

    def py_time(y):
        return scale(y, time_min, time_max, top + plot_h, top)

    parts = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2}" y="30" text-anchor="middle" font-size="22" font-family="Arial">MMBench Attention Sweep</text>'
    )
    parts.append(
        f'<text x="{width/2}" y="52" text-anchor="middle" font-size="14" font-family="Arial" fill="#555">{title}</text>'
    )

    x_ticks = rows
    for row in x_ticks:
        x_value = row["retained_ratio"] * 100.0
        x = px(x_value)
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top+plot_h}" stroke="#ececec" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top+plot_h+22}" text-anchor="middle" font-size="11" font-family="Arial">{format_x_tick_label(row)}</text>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top+plot_h+38}" text-anchor="middle" font-size="10" font-family="Arial" fill="#666">{row["avg_kept_vision_tokens"]:.0f} tok</text>'
        )

    y_ticks = 5
    for i in range(y_ticks + 1):
        acc_val = acc_min + (acc_max - acc_min) * i / y_ticks
        y = py_acc(acc_val)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" y2="{y:.2f}" stroke="#ececec" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y+4:.2f}" text-anchor="end" font-size="12" font-family="Arial" fill="#1f77b4">{acc_val:.3f}</text>'
        )

        time_val = time_min + (time_max - time_min) * i / y_ticks
        parts.append(
            f'<text x="{left+plot_w+12}" y="{y+4:.2f}" text-anchor="start" font-size="12" font-family="Arial" fill="#d62728">{time_val:.3f}</text>'
        )

    parts.append(
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{left+plot_w}" y1="{top}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333" stroke-width="1.5"/>'
    )

    acc_path = []
    time_path = []
    for idx, row in enumerate(rows):
        x = px(row["retained_ratio"] * 100.0)
        acc_y = py_acc(row["accuracy"])
        time_y = py_time(row["avg_generation_seconds"])
        cmd = "M" if idx == 0 else "L"
        acc_path.append(f"{cmd} {x:.2f} {acc_y:.2f}")
        time_path.append(f"{cmd} {x:.2f} {time_y:.2f}")

    parts.append(
        f'<path d="{" ".join(acc_path)}" fill="none" stroke="#1f77b4" stroke-width="3"/>'
    )
    parts.append(
        f'<path d="{" ".join(time_path)}" fill="none" stroke="#d62728" stroke-width="3"/>'
    )

    for row in rows:
        x = px(row["retained_ratio"] * 100.0)
        acc_y = py_acc(row["accuracy"])
        time_y = py_time(row["avg_generation_seconds"])
        parts.append(f'<circle cx="{x:.2f}" cy="{acc_y:.2f}" r="5" fill="#1f77b4"/>')
        parts.append(f'<circle cx="{x:.2f}" cy="{time_y:.2f}" r="5" fill="#d62728"/>')

    legend_x = left + 8
    legend_y = top + 10
    parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+26}" y2="{legend_y}" stroke="#1f77b4" stroke-width="3"/>')
    parts.append(f'<circle cx="{legend_x+13}" cy="{legend_y}" r="4.5" fill="#1f77b4"/>')
    parts.append(f'<text x="{legend_x+34}" y="{legend_y+4}" font-size="12" font-family="Arial">Accuracy</text>')

    legend_y_2 = legend_y + 20
    parts.append(f'<line x1="{legend_x}" y1="{legend_y_2}" x2="{legend_x+26}" y2="{legend_y_2}" stroke="#d62728" stroke-width="3"/>')
    parts.append(f'<circle cx="{legend_x+13}" cy="{legend_y_2}" r="4.5" fill="#d62728"/>')
    parts.append(f'<text x="{legend_x+34}" y="{legend_y_2+4}" font-size="12" font-family="Arial">Avg runtime (s)</text>')

    parts.append(
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="14" font-family="Arial">Percent of vision tokens retained</text>'
    )
    parts.append(
        f'<text x="24" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" fill="#1f77b4" transform="rotate(-90 24 {height/2})">Accuracy</text>'
    )
    parts.append(
        f'<text x="{width-24}" y="{height/2}" text-anchor="middle" font-size="14" font-family="Arial" fill="#d62728" transform="rotate(90 {width-24} {height/2})">Average runtime (seconds)</text>'
    )

    parts.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("playground/data/eval/mmbench/attention_sweep/mmbench_dev_20230712/summary.tsv"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("playground/data/eval/mmbench/attention_sweep/mmbench_dev_20230712/accuracy_time_vs_retained_tokens.svg"),
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Retained Vision Tokens vs Accuracy and Runtime",
    )
    args = parser.parse_args()

    rows = load_rows(args.summary_file)
    svg_plot(rows, args.output_file, args.title)
    print(args.output_file)


if __name__ == "__main__":
    main()
