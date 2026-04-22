#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path


def find_svg_files(input_paths):
    svg_files = []
    for path in input_paths:
        if path.is_dir():
            svg_files.extend(sorted(path.glob("*.svg")))
            continue
        if path.is_file() and path.suffix.lower() == ".svg":
            svg_files.append(path)
    return svg_files


def build_output_path(svg_path, output_dir):
    return output_dir / f"{svg_path.stem}.png"


def convert_svg(svg_path, output_path, zoom):
    cmd = [
        "rsvg-convert",
        "--zoom",
        str(zoom),
        str(svg_path),
        "--output",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert SVG patch visuals to PNG using rsvg-convert.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more SVG files or directories containing SVG files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for PNG outputs. Defaults to each SVG's current directory.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=2.0,
        help="Scale factor for rasterization. Higher values produce larger PNGs.",
    )
    args = parser.parse_args()

    svg_files = find_svg_files(args.inputs)
    if not svg_files:
        raise SystemExit("No SVG files found to convert.")

    converted = []
    for svg_path in svg_files:
        output_dir = args.output_dir or svg_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = build_output_path(svg_path, output_dir)
        convert_svg(svg_path, output_path, args.zoom)
        converted.append(output_path)

    for output_path in converted:
        print(f"Saved PNG to: {output_path}")


if __name__ == "__main__":
    main()
