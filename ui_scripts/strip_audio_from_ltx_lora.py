"""Remove audio-related tensors from an LTX LoRA safetensors file."""

import argparse
import json
import os
import sys
from collections import Counter

from safetensors import safe_open
from safetensors.torch import load_file, save_file


DEFAULT_PATTERNS = [
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "audio_to_video_attn",
    "video_to_audio_attn",
    "audio_",
    "_audio",
    "av_ca",
    "a2v",
    "v2a",
]


def should_drop_key(key: str, patterns: list[str]) -> bool:
    lowered = key.lower()
    return any(p.lower() in lowered for p in patterns)


def default_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}.no_audio{ext}"


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove audio-related tensors from a LoRA safetensors file."
    )
    parser.add_argument("--input_path", required=True, help="Path to the source LoRA safetensors file")
    parser.add_argument(
        "--output_path",
        default=None,
        help="Output path. Defaults to <input>.no_audio.safetensors",
    )
    parser.add_argument(
        "--contains",
        action="append",
        default=[],
        help="Additional substring to match for removal (repeatable).",
    )
    parser.add_argument(
        "--only_custom_patterns",
        action="store_true",
        help="Use only --contains patterns instead of the built-in LTX defaults.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report what would be removed without writing an output file.",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path) if args.output_path else default_output_path(input_path)

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr, flush=True)
        return 1

    patterns = list(args.contains)
    if not args.only_custom_patterns:
        patterns = DEFAULT_PATTERNS + patterns

    if not patterns:
        print("No removal patterns provided.", file=sys.stderr, flush=True)
        return 1

    log(f"Loading {input_path}")
    state_dict = load_file(input_path)

    with safe_open(input_path, framework="pt") as handle:
        metadata = handle.metadata() or {}

    removed_keys = [k for k in state_dict if should_drop_key(k, patterns)]
    kept_state_dict = {k: v for k, v in state_dict.items() if k not in removed_keys}

    log(f"Total tensors : {len(state_dict)}")
    log(f"Removed tensors: {len(removed_keys)}")
    log(f"Kept tensors  : {len(kept_state_dict)}")

    if removed_keys:
        counts: Counter = Counter()
        for key in removed_keys:
            for p in patterns:
                if p.lower() in key.lower():
                    counts[p] += 1
                    break
        log("Matched removal counts:")
        for p, n in counts.items():
            log(f"  {p}: {n}")
        log("Sample removed keys:")
        for key in removed_keys[:25]:
            log(f"  {key}")

    if args.dry_run:
        log("Dry run — no output file written.")
        print(json.dumps({"ok": True, "dry_run": True, "removed": len(removed_keys), "kept": len(kept_state_dict)}), flush=True)
        return 0

    updated_metadata = {
        **metadata,
        "repair_note": "Removed audio-related LoRA tensors using strip_audio_from_ltx_lora.py",
    }
    log(f"Saving to {output_path}")
    save_file(kept_state_dict, output_path, metadata=updated_metadata)
    log(f"Done.")

    print(
        json.dumps({
            "ok": True,
            "output": output_path,
            "removed": len(removed_keys),
            "kept": len(kept_state_dict),
        }),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
