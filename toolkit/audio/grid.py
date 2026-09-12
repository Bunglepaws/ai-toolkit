"""Duration-grid arithmetic for audio-only ("voice") training items.

Joint video+audio models quantize audio length to their video frame grid: the packed
sequence sizes its audio block from the *pixel* frame count, so a standalone audio file
can only train at the handful of durations that correspond to a legal frame count.

A model that supports voice items describes its grid by overriding
``BaseModel.get_audio_grid()``. This module holds only the arithmetic, so the model
stays the single source of truth for the constants (for minimax_h3 that is
``minimax_h3/src/packing.py``) and nothing under ``toolkit/`` has to import from an
extension.

The hop-exact trim is the subtle part. The audio VAE emits ``ceil(samples / hop)``
latents while the model demands ``round(frames / fps * latents_per_second)``. For the H3
grid those disagree by one at 56 and 107 frames, so audio is fitted to
``latents_for_frames(frames) * hop`` samples rather than to the nominal duration.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AudioGrid:
    """The legal durations for an audio-only item, and the sample math around them.

    frame_counts: legal pixel-frame counts, ascending. Excludes any count too short to
        be worth a training item (H3 drops its 5-frame slot: 0.208 s trains nothing).
    fps: frames per second the frame counts are expressed in.
    sample_rate: what the model's audio VAE consumes.
    latents_per_second: audio latent rate of that VAE.
    """

    frame_counts: Tuple[int, ...]
    fps: int
    sample_rate: int
    latents_per_second: int

    @property
    def hop_length(self) -> int:
        """Samples per audio latent."""
        return self.sample_rate // self.latents_per_second

    def latents_for_frames(self, frames: int) -> int:
        """Audio latents the model expects alongside `frames` pixel frames."""
        return int(round(frames / self.fps * self.latents_per_second))

    def hop_exact_samples(self, frames: int) -> int:
        """Sample count whose VAE encoding lands on exactly the latent count the model
        wants. This is the trim target -- NOT ``seconds_for_frames * sample_rate``,
        which is off by up to a hop."""
        return self.latents_for_frames(frames) * self.hop_length

    def seconds_for_frames(self, frames: int) -> float:
        """Nominal duration of a slot, for logging and error messages."""
        return frames / self.fps

    def snap_down(self, seconds: float, tolerance: float = 0.025) -> Optional[int]:
        """Largest legal frame count whose duration fits in `seconds`.

        Returns None when the audio is shorter than the smallest slot, i.e. there is no
        training item to be made from it. `tolerance` lets a file cut to the printed
        second-precision duration still claim its slot rather than falling to the one
        below (0.917 s of audio is meant to be the 22-frame slot, not a reject).
        """
        best = None
        for frames in self.frame_counts:
            if self.seconds_for_frames(frames) <= seconds + tolerance:
                best = frames
            else:
                break
        return best

    def durations_text(self) -> str:
        """'0.917, 1.625, ... or 5.167 seconds' -- for user-facing messages."""
        secs = [f"{self.seconds_for_frames(f):.3f}" for f in self.frame_counts]
        if len(secs) == 1:
            return f"{secs[0]} seconds"
        return ", ".join(secs[:-1]) + f" or {secs[-1]} seconds"


def fit_waveform_to_grid(waveform, frames: int, grid: AudioGrid):
    """Trim or zero-pad a (channels, samples) tensor to the hop-exact length for `frames`.

    Trims a long tail and pads a short one; both are sub-hop adjustments once the caller
    has already snapped the duration with ``snap_down``.
    """
    import torch

    want = grid.hop_exact_samples(frames)
    have = waveform.shape[-1]
    if have > want:
        return waveform[..., :want]
    if have < want:
        return torch.nn.functional.pad(waveform, (0, want - have))
    return waveform
