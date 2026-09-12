"""Generate voice training clips without launching a training job.

Exercises exactly the code path the trainer uses (toolkit.voice_clone.ensure_voice_clips),
so behaviour here is behaviour there -- including the manifest, so running it twice is a
no-op and --regenerate is the way to force a rebuild.

  python scripts/generate_voice_clips.py --reference ref.wav --target datasets/marc_voice
  python scripts/generate_voice_clips.py --config config/voice_test.yaml
  python scripts/generate_voice_clips.py --design "male, young adult, low pitch" --target ...
  python scripts/generate_voice_clips.py --target ... --dry-run

Needs OMNIVOICE_MODEL_PATH in the environment (the UI injects it from Settings).
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from toolkit.config_modules import VoiceCloneConfig  # noqa: E402
from toolkit.util.get_model import get_model_class  # noqa: E402
from toolkit.voice_clone import _pick_frame_counts, ensure_voice_clips  # noqa: E402
from toolkit.voice_clone.sentences import (  # noqa: E402
    DEFAULT_DIALOGUE,
    resolve_dialogue,
    validate_tags,
)


class _Arch:
    """get_model_class only reads .arch."""

    def __init__(self, arch):
        self.arch = arch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', help='job yaml; reads config.process[0].voice_clone')
    p.add_argument('--arch', default='minimax_h3', help='model arch supplying the audio grid')
    p.add_argument('--reference', help='reference audio/video (clone mode)')
    p.add_argument('--reference-text', default='', help='transcript; auto if blank')
    p.add_argument('--design', help='instruct string (design mode), e.g. "male, low pitch"')
    p.add_argument('--target', help='dataset folder to write clips into')
    p.add_argument('--seconds', type=float, default=60.0, help='total generated audio')
    p.add_argument('--mix', default='long', choices=['long', 'even', 'short'])
    p.add_argument('--description', default='a man speaking calmly, low pitch')
    p.add_argument('--trigger', default='')
    p.add_argument('--dialogue', help='text file, one line per utterance')
    p.add_argument('--backend', default='omnivoice')
    p.add_argument('--regenerate', action='store_true',
                   help='force a rebuild, deleting only clips this tool created')
    p.add_argument('--dry-run', action='store_true',
                   help='print the plan (clip count, durations, captions) and stop')
    args = p.parse_args()

    if args.config:
        from toolkit.config import get_config
        conf = get_config(args.config)
        raw = conf['config']['process'][0].get('voice_clone', {})
        cfg = VoiceCloneConfig(**raw)
        arch = conf['config']['process'][0].get('model', {}).get('arch', args.arch)
    else:
        if not args.target:
            p.error('--target is required without --config')
        dialogue = []
        if args.dialogue:
            with open(args.dialogue, encoding='utf-8') as f:
                dialogue = [l.strip() for l in f if l.strip()]
        cfg = VoiceCloneConfig(
            enabled=True,
            mode='design' if args.design else 'clone',
            reference_path=args.reference or '',
            reference_text=args.reference_text,
            instruct=args.design or '',
            target_dataset=args.target,
            target_seconds=args.seconds,
            duration_mix=args.mix,
            voice_description=args.description,
            trigger_word=args.trigger,
            dialogue=dialogue,
            backend=args.backend,
        )
        arch = args.arch

    if args.regenerate:
        cfg.regenerate_token = str(int(time.time()))

    grid = get_model_class(_Arch(arch)).get_audio_grid()
    if grid is None:
        p.error(f'{arch} does not support audio-only training items')

    if args.dry_run:
        # Generation is natural length now (see toolkit/voice_clone/backends.py) -- clip
        # count is still estimated from target_seconds, but per-clip duration is not
        # known until the model actually speaks the line, so this only shows the plan,
        # not a duration table.
        counts = _pick_frame_counts(grid, float(cfg.target_seconds), cfg.duration_mix)
        lines = resolve_dialogue(list(cfg.dialogue or []), len(counts))
        validate_tags(lines)
        print(f'{len(counts)} clip(s) planned (~{cfg.target_seconds:.0f}s target), '
              f'mix={cfg.duration_mix}')
        print(f'bank: {len(cfg.dialogue or DEFAULT_DIALOGUE)} line(s)')
        from toolkit.voice_clone import _caption
        for i, line in enumerate(lines, 1):
            print(f'  {i:>3}. {line}')
            print(f'       -> {_caption(line, cfg.voice_description, cfg.trigger_word)}')
        return

    ensure_voice_clips(cfg, grid)


if __name__ == '__main__':
    main()
