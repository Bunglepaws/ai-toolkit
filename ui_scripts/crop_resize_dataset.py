#!/usr/bin/env python3
"""Crop or resize dataset images to a target resolution bucket.

Streams NDJSON progress lines (one JSON object per line) to stdout.
Final line is {"type":"done",...}.
"""

import argparse
import json
import math
import os
import shutil
import sys

try:
    from PIL import Image
except ImportError:
    print(json.dumps({'type': 'error', 'message': 'PIL not available — install Pillow in the venv'}))
    sys.exit(1)

# Ensure TOOLKIT_ROOT is on sys.path so toolkit.buckets is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.buckets import get_bucket_for_image_size

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
CAPTION_EXTS = ['.txt', '.json', '.caption']


def resize_only(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Scale to fit within target WxH, preserving aspect ratio (no crop)."""
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def crop_to_exact(img: Image.Image, target_w: int, target_h: int,
                  anchor_x: float, anchor_y: float) -> Image.Image:
    """Resize then crop to exact WxH using anchor offset.

    anchor_x / anchor_y are in [0.0, 1.0]:
      0.0 = preserve the near edge (left / top)
      1.0 = preserve the far edge (right / bottom)
      0.5 = center crop (mnslarcher default)
    """
    w, h = img.size
    src_ar = w / h
    tgt_ar = target_w / target_h

    if src_ar > tgt_ar:
        # Wider than target — resize by height, crop excess width
        new_h = target_h
        new_w = math.ceil(w * new_h / h)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        excess = new_w - target_w
        left = int(round(excess * anchor_x))
        img = img.crop((left, 0, left + target_w, target_h))
    else:
        # Taller than target — resize by width, crop excess height
        new_w = target_w
        new_h = math.ceil(h * new_w / w)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        excess = new_h - target_h
        top = int(round(excess * anchor_y))
        img = img.crop((0, top, target_w, top + target_h))

    return img


def process(input_dir: str, output_dir: str, resolution: int, divisibility: int,
            mode: str, decisions: dict) -> None:
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                image_files.append(os.path.join(root, fname))

    total = len(image_files)
    skipped = 0

    for i, src_path in enumerate(image_files):
        fname = os.path.basename(src_path)
        try:
            img = Image.open(src_path)
            orig_format = img.format or 'PNG'

            w, h = img.size
            bucket = get_bucket_for_image_size(w, h, resolution, divisibility)
            bw, bh = bucket['width'], bucket['height']

            if mode == 'crop':
                dec = decisions.get(src_path, {})
                ax = float(dec.get('anchor_x', 0.5))
                ay = float(dec.get('anchor_y', 0.5))
                out_img = crop_to_exact(img, bw, bh, ax, ay)
            else:
                out_img = resize_only(img, bw, bh)

            ext = os.path.splitext(fname)[1].lower()
            dst_path = os.path.join(output_dir, fname)

            if ext in ('.jpg', '.jpeg'):
                if out_img.mode in ('RGBA', 'P', 'LA'):
                    out_img = out_img.convert('RGB')
                out_img.save(dst_path, 'JPEG', quality=95)
            elif ext == '.webp':
                out_img.save(dst_path, 'WEBP', quality=95)
            else:
                # PNG — preserve mode as-is where possible
                out_img.save(dst_path, 'PNG')

            # Copy all caption sidecars alongside the processed image
            stem = os.path.splitext(src_path)[0]
            dst_stem = os.path.join(output_dir, os.path.splitext(fname)[0])
            for cap_ext in CAPTION_EXTS:
                cap_src = stem + cap_ext
                if os.path.exists(cap_src):
                    shutil.copy2(cap_src, dst_stem + cap_ext)

            print(json.dumps({'type': 'progress', 'current': i + 1, 'total': total, 'name': fname}),
                  flush=True)

        except Exception as e:
            skipped += 1
            print(json.dumps({'type': 'warning', 'name': fname, 'error': str(e)}), flush=True)

    print(json.dumps({'type': 'done', 'output_dir': output_dir, 'total': total, 'skipped': skipped}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description='Crop/resize dataset images for bucket consolidation')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--divisibility', type=int, default=8)
    parser.add_argument('--mode', choices=['resize', 'crop'], default='resize')
    parser.add_argument('--decisions', default='', help='Path to JSON file with per-image anchor decisions')
    args = parser.parse_args()

    decisions: dict = {}
    if args.decisions and os.path.exists(args.decisions):
        with open(args.decisions, 'r', encoding='utf-8') as f:
            decisions = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    process(args.input_dir, args.output_dir, args.resolution, args.divisibility,
            args.mode, decisions)


if __name__ == '__main__':
    main()
