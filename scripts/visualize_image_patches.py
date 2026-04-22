#!/usr/bin/env python3

import argparse
import base64
import json
import struct
from pathlib import Path


def xml_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def get_png_size(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Invalid PNG header.")
    width, height = struct.unpack(">II", data[16:24])
    return width, height, "png"


def get_gif_size(data):
    if data[:6] not in (b"GIF87a", b"GIF89a"):
        raise ValueError("Invalid GIF header.")
    width, height = struct.unpack("<HH", data[6:10])
    return width, height, "gif"


def get_jpeg_size(data):
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("Invalid JPEG header.")

    offset = 2
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }

    while offset < len(data):
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break

        marker = data[offset]
        offset += 1

        if marker in (0xD8, 0xD9):
            continue
        if marker == 0xDA:
            break
        if offset + 2 > len(data):
            break

        segment_length = struct.unpack(">H", data[offset:offset + 2])[0]
        if segment_length < 2:
            break

        if marker in sof_markers:
            if offset + 7 > len(data):
                break
            height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
            return width, height, "jpeg"

        offset += segment_length

    raise ValueError("Could not determine JPEG size.")


def get_image_info(image_path):
    data = image_path.read_bytes()
    if data.startswith(b"\xff\xd8"):
        width, height, kind = get_jpeg_size(data)
    elif data.startswith(b"\x89PNG\r\n\x1a\n"):
        width, height, kind = get_png_size(data)
    elif data.startswith((b"GIF87a", b"GIF89a")):
        width, height, kind = get_gif_size(data)
    else:
        raise ValueError(f"Unsupported image format for {image_path}.")
    return {
        "bytes": data,
        "width": width,
        "height": height,
        "mime_type": f"image/{kind}",
    }


def make_data_uri(image_bytes, mime_type):
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def write_grid_overlay_svg(output_path, image_href, width, height, rows, cols):
    patch_w = width / cols
    patch_h = height / rows
    label_h = 36
    svg_h = height + label_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}" viewBox="0 0 {width} {svg_h}">',
        '<rect width="100%" height="100%" fill="#f7f3ea"/>',
        f'<image href="{image_href}" x="0" y="{label_h}" width="{width}" height="{height}"/>',
    ]

    for col in range(1, cols):
        x = patch_w * col
        parts.append(
            f'<line x1="{x}" y1="{label_h}" x2="{x}" y2="{label_h + height}" stroke="#ef4444" stroke-width="2"/>'
        )
    for row in range(1, rows):
        y = label_h + patch_h * row
        parts.append(
            f'<line x1="0" y1="{y}" x2="{width}" y2="{y}" stroke="#ef4444" stroke-width="2"/>'
        )

    patch_index = 0
    for row in range(rows):
        for col in range(cols):
            x = patch_w * col
            y = label_h + patch_h * row
            parts.append(
                f'<rect x="{x + 4}" y="{y + 4}" width="36" height="20" rx="4" fill="#111827" opacity="0.85"/>'
            )
            parts.append(
                f'<text x="{x + 22}" y="{y + 18}" font-family="monospace" font-size="12" text-anchor="middle" fill="#ffffff">P{patch_index}</text>'
            )
            patch_index += 1

    parts.append(
        '<text x="12" y="24" font-family="monospace" font-size="18" fill="#111827">3x3 patch grid overlay</text>'
    )
    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_patch_sheet_svg(output_path, image_href, width, height, rows, cols, gap=16, label_h=28):
    patch_w = width / cols
    patch_h = height / rows
    sheet_w = cols * patch_w + (cols + 1) * gap
    sheet_h = rows * (patch_h + label_h) + (rows + 1) * gap

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sheet_w}" height="{sheet_h}" viewBox="0 0 {sheet_w} {sheet_h}">',
        '<rect width="100%" height="100%" fill="#f7f3ea"/>',
    ]

    patch_index = 0
    for row in range(rows):
        for col in range(cols):
            x = gap + col * (patch_w + gap)
            y = gap + row * (patch_h + label_h + gap)
            vx = col * patch_w
            vy = row * patch_h

            parts.append(f'<g transform="translate({x},{y})">')
            parts.append(
                f'<rect x="0" y="0" width="{patch_w}" height="{patch_h}" fill="#ffffff" stroke="#d1d5db" stroke-width="1.5"/>'
            )
            parts.append(
                f'<svg x="0" y="0" width="{patch_w}" height="{patch_h}" viewBox="{vx} {vy} {patch_w} {patch_h}" overflow="hidden">'
            )
            parts.append(f'<image href="{image_href}" x="0" y="0" width="{width}" height="{height}"/>')
            parts.append("</svg>")
            parts.append(
                f'<text x="{patch_w / 2}" y="{patch_h + 18}" font-family="monospace" font-size="14" text-anchor="middle" fill="#111827">P{patch_index}</text>'
            )
            parts.append("</g>")
            patch_index += 1

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_metadata(output_path, image_path, width, height, rows, cols):
    patch_w = width / cols
    patch_h = height / rows
    patches = []
    patch_index = 0
    for row in range(rows):
        for col in range(cols):
            patches.append(
                {
                    "patch_id": f"P{patch_index}",
                    "row": row,
                    "col": col,
                    "x": patch_w * col,
                    "y": patch_h * row,
                    "width": patch_w,
                    "height": patch_h,
                }
            )
            patch_index += 1

    payload = {
        "source_image": str(image_path),
        "image_width": width,
        "image_height": height,
        "rows": rows,
        "cols": cols,
        "patches": patches,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create a simple patching visual for an image.")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Source image path.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=3,
        help="Number of rows of patches.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns of patches.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("playground/data/eval/mmbench/extracted_samples"),
        help="Where to write the SVG outputs.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Output filename prefix. Defaults to the source image stem plus patch dimensions.",
    )
    args = parser.parse_args()

    image_info = get_image_info(args.image)
    image_href = make_data_uri(image_info["bytes"], image_info["mime_type"])
    prefix = args.prefix or f"{args.image.stem}_{args.rows}x{args.cols}"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = args.output_dir / f"{prefix}_overlay.svg"
    sheet_path = args.output_dir / f"{prefix}_patches.svg"
    metadata_path = args.output_dir / f"{prefix}_patches.json"

    write_grid_overlay_svg(
        overlay_path,
        image_href,
        image_info["width"],
        image_info["height"],
        args.rows,
        args.cols,
    )
    write_patch_sheet_svg(
        sheet_path,
        image_href,
        image_info["width"],
        image_info["height"],
        args.rows,
        args.cols,
    )
    write_metadata(
        metadata_path,
        args.image,
        image_info["width"],
        image_info["height"],
        args.rows,
        args.cols,
    )

    print(f"Saved overlay SVG to: {overlay_path}")
    print(f"Saved patch sheet SVG to: {sheet_path}")
    print(f"Saved patch metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
