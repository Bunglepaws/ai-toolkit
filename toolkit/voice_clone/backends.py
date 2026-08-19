"""TTS backends for voice-clip generation.

One method, so another engine (Qwen Voice, Fish S2, ...) slots in by adding a class and a
registry entry -- nothing in the generation flow changes.

Three measured facts about OmniVoice shape this (verified 2026-08-18, v0.2.1):

  * `duration` lands within one TTS token, not exactly. OmniVoice quantizes to its own
    ~40 ms grid and rounds DOWN -- asking for 5.175 s returned 5.160 s on every clip.
    That removes the generate-measure-snap-discard SEARCH, but the caller's hop-exact
    pad is still load-bearing, not a rounding guard.
  * `postprocess_output=True` is free here. The docs warn it trims trailing silence and
    shortens output; measured, with `duration` set the model fills the time with speech
    so there is nothing to trim -- output was identical either way. Leave it and
    `denoise` on: an artefact in a training corpus repeats across every clip instead of
    averaging out.
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
        durations: List[float],
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        seed: int = 42,
    ) -> List[np.ndarray]:
        """One waveform per text, each `durations[i]` seconds (to within one TTS token).

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

    def generate(
        self,
        texts: List[str],
        durations: List[float],
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
        for i, (text, dur) in enumerate(zip(texts, durations)):
            # seed+i: reproducible run, but a repeated line still gets its own take
            torch.manual_seed(seed + i)
            audio = self.model.generate(
                text=text,
                duration=float(dur),
                # Cleaner speech. Worth having in training data -- any artefact here is
                # reproduced identically across every clip rather than averaging out.
                denoise=True,
                # Strips dead air. This makes the output shorter than `duration` asks
                # for, which would be fatal if we needed exact lengths -- but we don't:
                # OmniVoice quantizes to its own ~40 ms token grid anyway and always
                # lands short, so the loader's hop-exact pad is load-bearing either way.
                # Given that, trimming dead air and padding the tail beats training on
                # the model's own pauses.
                postprocess_output=True,
                # cleans the REFERENCE: strips its long silences and punctuates the
                # transcript. Matters for a 20 s recording used as a voice prompt.
                preprocess_prompt=True,
                # 0.1 s of silence per side is 22% of the shortest slot
                pad_duration=0.0,
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
