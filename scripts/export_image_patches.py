#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

from visualize_image_patches import get_image_info, make_data_uri


def write_patch_svg(output_path, image_href, image_width, image_height, x, y, patch_width, patch_height):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{patch_width}" height="{patch_height}" viewBox="{x} {y} {patch_width} {patch_height}">',
        f'<image href="{image_href}" x="0" y="0" width="{image_width}" height="{image_height}"/>',
        "</svg>",
    ]
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def export_patch_svgs(image_path, output_dir, rows, cols, prefix):
    image_info = get_image_info(image_path)
    image_href = make_data_uri(image_info["bytes"], image_info["mime_type"])
    patch_width = image_info["width"] / cols
    patch_height = image_info["height"] / rows

    output_dir.mkdir(parents=True, exist_ok=True)

    svg_paths = []
    patch_index = 0
    for row in range(rows):
        for col in range(cols):
            x = col * patch_width
            y = row * patch_height
            svg_path = output_dir / f"{prefix}_P{patch_index}.svg"
            write_patch_svg(
                svg_path,
                image_href,
                image_info["width"],
                image_info["height"],
                x,
                y,
                patch_width,
                patch_height,
            )
            svg_paths.append(svg_path)
            patch_index += 1

    return svg_paths


def convert_svgs_to_png(svg_paths, zoom):
    png_paths = []
    for svg_path in svg_paths:
        png_path = svg_path.with_suffix(".png")
        cmd = [
            "rsvg-convert",
            "--zoom",
            str(zoom),
            str(svg_path),
            "--output",
            str(png_path),
        ]
        subprocess.run(cmd, check=True)
        png_paths.append(png_path)
    return png_paths


def main():
    parser = argparse.ArgumentParser(description="Export an image into standalone patch images.")
    parser.add_argument("--image", type=Path, required=True, help="Source image path.")
    parser.add_argument("--rows", type=int, default=3, help="Number of patch rows.")
    parser.add_argument("--cols", type=int, default=3, help="Number of patch columns.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("playground/data/eval/mmbench/extracted_samples"),
        help="Directory for exported patch files.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output prefix. Defaults to the image stem plus patch dimensions.",
    )
    parser.add_argument(
        "--png",
        action="store_true",
        help="Also render each patch SVG to a PNG using rsvg-convert.",
    )
    parser.add_argument(
        "--zoom",
        type=float,
        default=3.0,
        help="PNG rasterization zoom factor when --png is set.",
    )
    args = parser.parse_args()

    prefix = args.prefix or f"{args.image.stem}_{args.rows}x{args.cols}"
    svg_paths = export_patch_svgs(args.image, args.output_dir, args.rows, args.cols, prefix)

    for svg_path in svg_paths:
        print(f"Saved patch SVG to: {svg_path}")

    if args.png:
        png_paths = convert_svgs_to_png(svg_paths, args.zoom)
        for png_path in png_paths:
            print(f"Saved patch PNG to: {png_path}")


if __name__ == "__main__":
    main()
