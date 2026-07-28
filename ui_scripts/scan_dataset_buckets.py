#!/usr/bin/env python3
"""Scan dataset images, compute resolution buckets, and return a JSON summary.

Emits a single JSON object on the last stdout line — compatible with the
parseResult() helper in /api/scripts/route.ts.
"""

import argparse
import json
import os
import sys

try:
    from PIL import Image
except ImportError:
    print(json.dumps({'error': 'PIL not available — install Pillow in the venv'}))
    sys.exit(1)

# Ensure TOOLKIT_ROOT is on sys.path so toolkit.buckets is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from toolkit.buckets import get_bucket_for_image_size

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}


def scan(dataset_dir: str, resolution: int, divisibility: int) -> dict:
    images = []
    for root, dirs, files in os.walk(dataset_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for fname in sorted(files):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in IMAGE_EXTS:
                continue
            fpath = os.path.join(root, fname)
            try:
                with Image.open(fpath) as img:
                    w, h = img.size
            except Exception as e:
                print(f'[skip] {fname}: {e}', file=sys.stderr)
                continue

            bucket = get_bucket_for_image_size(w, h, resolution, divisibility)
            bw, bh = bucket['width'], bucket['height']

            # needs_crop when the image AR differs from the bucket AR by >2%
            src_ar = w / h
            bkt_ar = bw / bh
            needs_crop = abs(src_ar - bkt_ar) / bkt_ar > 0.02

            images.append({
                'path': fpath,
                'width': w,
                'height': h,
                'bucket_w': bw,
                'bucket_h': bh,
                'needs_crop': needs_crop,
            })

    buckets: dict[str, int] = {}
    for img in images:
        key = f"{img['bucket_w']}x{img['bucket_h']}"
        buckets[key] = buckets.get(key, 0) + 1

    return {'images': images, 'buckets': buckets}


def main():
    parser = argparse.ArgumentParser(description='Scan dataset image buckets')
    parser.add_argument('--dataset-dir', required=True, help='Absolute path to dataset folder')
    parser.add_argument('--resolution', type=int, default=1024, help='Target pixel budget side length')
    parser.add_argument('--divisibility', type=int, default=8, help='Bucket grid alignment (8 for ai-toolkit)')
    args = parser.parse_args()

    if not os.path.isdir(args.dataset_dir):
        print(json.dumps({'error': f'Directory not found: {args.dataset_dir}'}))
        sys.exit(1)

    result = scan(args.dataset_dir, args.resolution, args.divisibility)
    # Print as the last line so parseResult() can find it
    print(json.dumps(result))


if __name__ == '__main__':
    main()
