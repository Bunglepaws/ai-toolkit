"""TTS backends for voice-clip generation.

One method, so another engine (Qwen Voice, Fish S2, ...) slots in by adding a class and a
registry entry -- nothing in the generation flow changes.

**Generation is NATURAL LENGTH, not forced.** An earlier version passed `duration=` to
land each clip on an exact H3 grid slot. Measured 2026-08-21: forcing duration on a line
whose natural length is shorter makes the model fill the gap by repeating words, stuttering,
or over-emphasising -- audible defects, not edge cases (3 of 12 clips in one real run).
Comparing the same three lines with `speed=0.9` instead of a forced duration, all three came
out clean. So generation now asks for natural delivery and the RESULT is fit to the grid
afterwards -- the dataset loader already re-derives a clip's frame count from its actual
on-disk duration (see toolkit/dataloader_mixins.py's voice-item path), so nothing downstream
needed to change for this.

Other measured facts (verified 2026-08-18/21, v0.2.1):

  * `postprocess_output=True` is free. Docs warn it trims trailing silence and shortens
    output; measured, output was the same length either way. Leave it and `denoise` on:
    an artefact in a training corpus repeats across every clip instead of averaging out.
  * Generation is stochastic. Two unseeded runs of the same text differ (max sample
    delta 0.92), so a cycled dialogue line genuinely gets a different take. Seeding via
    `torch.manual_seed` makes that reproducible without making it uniform.

`instruct` designs a voice from attributes with no reference audio, but nothing
guarantees two calls with the same instruct string produce the SAME voice -- so design
mode generates one seed utterance and then CLONES from it. See ensure_voice_clips.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class TTSBackend(ABC):
    """sample_rate is what `clone`/`design` return, not what the trainer wants -- the
    caller writes the wav at this rate and the model's own encode_audio resamples."""

    sample_rate: int = 24000

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def generate(
        self,
        texts: List[str],
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """One waveform per text, at whatever length the voice naturally delivers it.

        No duration is requested -- see the module docstring for why forcing one produces
        audible repetition/stutter/over-emphasis. The caller fits the result to the H3 grid
        afterwards; it does not need to know the length in advance.

        Exactly one of `ref_audio` (clone) or `instruct` (design) is given.

        `seed` is applied per clip as seed+i. Generation is stochastic by default --
        measured: two unseeded runs of the same text differ (max sample delta 0.92) --
        so seeding makes runs reproducible while still giving a cycled dialogue line a
        genuinely different take each time it comes round.
        """

    def unload(self) -> None:
        """Release the model. Called before the diffusion model loads, so a voice step
        never overlaps the trainer in VRAM."""
        pass


class OmniVoiceBackend(TTSBackend):
    name = "omnivoice"
    sample_rate = 24000

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda:0"):
        # Path comes from the OMNIVOICE_MODEL_PATH global setting, injected into the
        # training process env. Falling back to the HF repo id would silently start a
        # multi-GB download, so an unset path is an error the caller raises, not a default.
        self.model_path = model_path or os.environ.get("OMNIVOICE_MODEL_PATH")
        self.device = device
        self.model = None
        self._prompt = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path:
            raise ValueError(
                "No OmniVoice model path. Set OMNIVOICE_MODEL_PATH in the environment, or "
                "the 'OmniVoice model path' row in Settings."
            )
        if not os.path.exists(self.model_path):
            raise ValueError(f"OmniVoice model path does not exist: {self.model_path}")
        try:
            import torch
            from omnivoice import OmniVoice  # lazy: a missing package must not break
        except ImportError as e:                # unrelated model discovery
            raise ImportError(
                f"omnivoice is not installed ({e}). pip install omnivoice"
            ) from e
        self.model = OmniVoice.from_pretrained(
            self.model_path, device_map=self.device, dtype=torch.bfloat16
        )

    # Marc's ComfyUI workflow uses 0.9; adopted as the default here after measuring that
    # forcing an exact duration (the previous approach) makes the model fill short lines
    # with repeated words, stutters, or over-emphasis to hit the target length.
    NATURAL_SPEED = 0.9

    def generate(
        self,
        texts: List[str],
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        seed: int = 42,
    ) -> List[np.ndarray]:
        import torch

        self.load()
        if (ref_audio is None) == (instruct is None):
            raise ValueError("give exactly one of ref_audio (clone) or instruct (design)")

        kwargs = {}
        if ref_audio is not None:
            # Encode the reference ONCE and reuse it. Re-encoding per clip is pure waste.
            if self._prompt is None and hasattr(self.model, "create_voice_clone_prompt"):
                try:
                    self._prompt = self.model.create_voice_clone_prompt(
                        ref_audio=ref_audio, ref_text=ref_text or None
                    )
                except Exception:
                    self._prompt = None      # fall back to per-call ref_audio
            if self._prompt is not None:
                kwargs["voice_clone_prompt"] = self._prompt
            else:
                kwargs["ref_audio"] = ref_audio
                if ref_text:
                    kwargs["ref_text"] = ref_text
        else:
            kwargs["instruct"] = instruct

        out = []
        for i, text in enumerate(texts):
            # seed+i: reproducible run, but a repeated line still gets its own take
            torch.manual_seed(seed + i)
            audio = self.model.generate(
                text=text,
                # NATURAL length -- no `duration=`. See NATURAL_SPEED above and the
                # module docstring for why forcing a target length backfired.
                speed=self.NATURAL_SPEED,
                # Cleaner speech. Worth having in training data -- any artefact here is
                # reproduced identically across every clip rather than averaging out.
                denoise=True,
                # Strips dead air.
                postprocess_output=True,
                # cleans the REFERENCE: strips its long silences and punctuates the
                # transcript.
                preprocess_prompt=True,
                # keep a hair of fade so clip edges don't click
                fade_duration=0.01,
                **kwargs,
            )
            wav = audio[0] if isinstance(audio, (list, tuple)) else audio
            out.append(np.asarray(wav, dtype=np.float32).reshape(-1))
        return out

    def unload(self) -> None:
        self.model = None
        self._prompt = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


BACKENDS = {OmniVoiceBackend.name: OmniVoiceBackend}


def get_backend(name: str, **kwargs) -> TTSBackend:
    if name not in BACKENDS:
        raise ValueError(
            f"Unknown TTS backend '{name}'. Available: {', '.join(sorted(BACKENDS))}"
        )
    return BACKENDS[name](**kwargs)
