"""Generate voice training clips from a reference recording or a voice description.

Entry point is `ensure_voice_clips(config, audio_grid)`, called from the trainer's
`hook_before_model_load` -- the first hook in run(), before the diffusion model is even
resolved. Three reasons that seam and not a later one:

  1. A bad reference path or a missing package fails in SECONDS, not after the minutes
     H3 spends on quantization and the text encoder.
  2. The TTS model loads, generates, and is fully released before the trainer touches the
     GPU. Zero VRAM overlap, so this cannot make an already-tight job OOM.
  3. The clips are on disk before the dataloader scans the folder, so they are picked up
     as ordinary voice items with no special-casing.

It is a no-op on every run after the first -- see manifest.py.
"""

import os
from typing import List, Optional

from toolkit.audio.grid import AudioGrid
from toolkit.voice_clone import manifest as _manifest
from toolkit.voice_clone.backends import get_backend
from toolkit.voice_clone.sentences import (
    DEFAULT_DIALOGUE,
    resolve_dialogue,
    split_tags,
    validate_tags,
)

# Fraction of clips at each slot, from the longest down. 'long' (the default) puts
# everything in the top slot: a 5.167 s clip carries 207 audio latents against 37 for a
# 0.917 s one -- 5.6x the voice content -- while its video placeholder grows only from 7
# rows to 37. The others exist for experiments.
DURATION_MIXES = {
    "long": [(0, 1.0)],                               # 0 = longest slot
    "even": None,                                     # spread across all slots
    "short": [(-1, 0.4), (-2, 0.35), (-3, 0.25)],     # bottom three
}


def _pick_frame_counts(grid: AudioGrid, target_seconds: float, mix: str) -> List[int]:
    """Frame counts for the clips, totalling at or just above target_seconds."""
    slots = list(grid.frame_counts)
    if mix == "even":
        order = list(reversed(slots))
        out, total, i = [], 0.0, 0
        while total < target_seconds:
            f = order[i % len(order)]
            out.append(f)
            total += grid.seconds_for_frames(f)
            i += 1
        return out

    weights = DURATION_MIXES.get(mix) or DURATION_MIXES["long"]
    order = list(reversed(slots))  # longest first
    out, total, i = [], 0.0, 0
    pattern = []
    for idx, w in weights:
        f = order[idx] if idx >= 0 else slots[idx]
        pattern.extend([f] * max(1, int(round(w * 10))))
    while total < target_seconds:
        f = pattern[i % len(pattern)]
        out.append(f)
        total += grid.seconds_for_frames(f)
        i += 1
    return out


def _write_wav(path: str, wav, sample_rate: int) -> None:
    import numpy as np
    import torch
    import torchaudio

    t = torch.from_numpy(np.asarray(wav, dtype="float32")).reshape(1, -1)
    torchaudio.save(path, t, sample_rate)


def _caption(line: str, voice_description: str, trigger_word: str) -> str:
    """Caption the VOICE, not a picture, with the spoken words appended.

    Non-verbal tags are stripped and replaced with plain language: `[laughter]` means
    something to the TTS and nothing to a diffusion model's text encoder, which would read
    it as literal tokens.
    """
    phrases, spoken = split_tags(line)
    bits = []
    if trigger_word:
        # Write the ``[trigger]`` PLACEHOLDER, not the resolved word. That is the
        # convention the image datasets already use, and the dataset's own
        # trigger_word substitutes it at caption-load time -- so changing the job's
        # trigger does not leave a stale name baked into every voice caption on disk.
        bits.append("[trigger]")
    if voice_description:
        bits.append(voice_description.strip())
    bits.extend(phrases)
    head = ", ".join(b for b in bits if b)
    if spoken:
        return f'{head}, saying "{spoken}"' if head else f'saying "{spoken}"'
    return head


def ensure_voice_clips(config, audio_grid: AudioGrid, print_fn=print) -> Optional[dict]:
    """Generate the clips if needed. Returns a summary dict, or None when disabled."""
    if config is None or not getattr(config, "enabled", False):
        return None

    target = config.target_dataset
    if not target:
        raise ValueError("voice_clone.target_dataset is not set")
    os.makedirs(target, exist_ok=True)

    frame_counts = _pick_frame_counts(
        audio_grid, float(config.target_seconds), config.duration_mix
    )
    lines = resolve_dialogue(list(config.dialogue or []), len(frame_counts))
    validate_tags(lines)

    supplied = [l for l in (config.dialogue or DEFAULT_DIALOGUE) if l and l.strip()]
    if len(supplied) < len(frame_counts):
        print_fn(
            f"Voice clone: {len(supplied)} dialogue line(s) supplied, "
            f"{len(frame_counts)} clip(s) requested, cycling "
            f"{len(frame_counts) / len(supplied):.1f}x"
        )

    # Fingerprint the SUPPLIED lines, not the cycled ones: the cycled list length tracks
    # the clip count, so hashing it would make any target_seconds change also report
    # "dialogue lines" as changed and muddy the error message.
    fingerprint = _manifest.build_fingerprint(config, supplied)
    action, existing = _manifest.decide(
        target, fingerprint, getattr(config, "regenerate_token", "")
    )

    if action == "skip":
        print_fn(
            f"Voice clone: {len(existing.get('files', []))} clip(s) already generated in "
            f"{target}, settings unchanged -- skipping."
        )
        return {"action": "skip", "files": existing.get("files", [])}

    if action == "regenerate":
        n = _manifest.delete_generated(target, existing)
        print_fn(f"Voice clone: regenerate requested, removed {n} previous clip(s).")

    total_s = sum(audio_grid.seconds_for_frames(f) for f in frame_counts)
    print_fn(
        f"Voice clone: generating {len(frame_counts)} clip(s), {total_s:.1f}s total, "
        f"via {config.backend} ({config.mode} mode)"
    )

    backend = get_backend(config.backend)
    written: List[str] = []
    try:
        ref_audio = config.reference_path or None
        ref_text = config.reference_text or None

        if config.mode == "design":
            # Design mode must bootstrap into clone mode. Nothing guarantees two calls
            # with the same instruct string give the SAME voice, and there is no seed --
            # so a dataset generated straight from `instruct` risks being N different
            # speakers matching one description, which averages into mush. Instead:
            # generate ONE seed utterance, then clone the rest from it. Correct whether
            # or not design mode turns out to be voice-stable.
            seed_path = config.voice_seed_path or os.path.join(target, ".voice_seed.wav")
            if not os.path.exists(seed_path):
                if not config.instruct:
                    raise ValueError("design mode needs an 'instruct' string")
                print_fn(f"Voice clone: designing a seed voice -- {config.instruct}")
                seed_wav = backend.generate(
                    [lines[0]],
                    [audio_grid.seconds_for_frames(frame_counts[0])],
                    instruct=config.instruct,
                )[0]
                _write_wav(seed_path, seed_wav, backend.sample_rate)
                print_fn(f"Voice clone: seed voice written to {seed_path}")
            ref_audio, ref_text = seed_path, lines[0]

        if not ref_audio:
            raise ValueError("clone mode needs a reference_path")
        if not os.path.exists(ref_audio):
            raise ValueError(f"reference audio not found: {ref_audio}")

        # Ask for the HOP-EXACT duration, not the nominal one: for 124 frames the model
        # wants 207*800 = 165600 samples (5.175s) while 124/24 is 5.1667s. Requesting the
        # hop-exact length lands the sample count right and leaves the dataloader's fit
        # as a few-sample guard rather than a real cut.
        durations = [
            audio_grid.hop_exact_samples(f) / audio_grid.sample_rate for f in frame_counts
        ]
        wavs = backend.generate(
            lines, durations, ref_audio=ref_audio, ref_text=ref_text,
            seed=int(getattr(config, 'seed', 42)),
        )

        hist = {}
        for i, (wav, frames) in enumerate(zip(wavs, frame_counts)):
            name = f"voice_{i + 1:04d}.wav"
            _write_wav(os.path.join(target, name), wav, backend.sample_rate)
            with open(
                os.path.join(target, os.path.splitext(name)[0] + ".txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(_caption(lines[i], config.voice_description, config.trigger_word))
            written.append(name)
            hist[frames] = hist.get(frames, 0) + 1
    finally:
        # Release before the diffusion model loads, whatever happened.
        backend.unload()

    _manifest.write(
        target, fingerprint, written, getattr(config, "regenerate_token", "")
    )
    print_fn(
        "Voice clone: wrote "
        + ", ".join(
            f"{n}x{audio_grid.seconds_for_frames(f):.3f}s" for f, n in sorted(hist.items())
        )
        + f" -> {target}"
    )
    return {"action": action, "files": written, "total_seconds": total_s}
