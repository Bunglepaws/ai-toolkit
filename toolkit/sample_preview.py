"""Live previews of the clip being sampled.

Sampling a video is slow enough that a still progress bar is indistinguishable
from a hung job -- which is exactly how it has been read. This decodes the
sampler's running x0 estimate through a tiny VAE and writes it to the job folder
as a short looping MP4, so "is this working" is answerable by looking at it.

Cost, measured on MiniMax-H3 at 768: 625 ms to decode 120 frames, ~150 ms to
encode them. An H3 denoise step is tens of seconds, so a preview per step is
low single-digit percent. The decode is skipped entirely when previews are off.

Files written to the job folder, both replaced atomically:

    preview.mp4     the clip so far, looping
    preview.json    {sample, of, step, total, updated, file}

Configured by environment. AITK_SAMPLE_PREVIEW has a Settings page checkbox
(ui/src/app/settings/page.tsx) and is injected by the worker
(ui/cron/actions/startJob.ts); the rest are env-only -- set them in the shell
that launches the UI/worker:

    AITK_SAMPLE_PREVIEW           "0" disables everything here (default on)
    AITK_SAMPLE_PREVIEW_FRAMES    frames in the preview clip (default 60)
    AITK_SAMPLE_PREVIEW_FPS       playback rate (default 12)
    AITK_SAMPLE_PREVIEW_MAX_RES   longest side cap, 0 = native (default 1024)
"""

import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch

# arch -> tiny VAE filename under <MODELS_PATH>/vae_approx.
# Only TAEHV-family (video) checkpoints belong here; image models use the TAESD
# layout, which load_taehv_decoder refuses outright rather than half-loading.
#
# LTX-2.5 deliberately excluded: its README describes a new diffusion video
# decoder, and no taeltx checkpoint has been verified against its latent
# geometry. Guessing wrong here means a plausible-looking but wrong preview,
# not a loud failure -- worse than no preview at all.
#
# ltx2 / ltx2.3's decode path (extensions_built_in/diffusion_models/ltx2/ltx2.py,
# _stop_callback) is built from reading diffusers' pipeline_ltx2*.py source and
# cross-checked against kijai's ComfyUI-LTXVideo preview node, not verified
# against a live run -- watch the first real preview for a plausible (even if
# blurry) image forming, not noise or static colour, before trusting it.
TINY_VAE_BY_ARCH = {
    'minimax_h3': 'taeh3.safetensors',
    'ltx2': 'taeltx_2.safetensors',
    'ltx2.3': 'taeltx2_3.safetensors',
}


def _env_int(name, default):
    try:
        return int(os.environ.get(name, '') or default)
    except ValueError:
        return default


class SamplePreviewWriter:
    """Decodes and writes previews for one sampling run. Never raises into the
    sampler: a preview is a convenience, and losing it must not cost a sample."""

    def __init__(self, arch, job_folder, print_fn=print):
        self.arch = arch
        self.job_folder = job_folder
        self.print_fn = print_fn
        self.enabled = (
            os.environ.get('AITK_SAMPLE_PREVIEW', '1') == '1'
            and arch in TINY_VAE_BY_ARCH
            and job_folder is not None
        )
        self.frames = _env_int('AITK_SAMPLE_PREVIEW_FRAMES', 60)
        self.fps = max(1, _env_int('AITK_SAMPLE_PREVIEW_FPS', 12))
        self.max_res = _env_int('AITK_SAMPLE_PREVIEW_MAX_RES', 1024)
        self.decoder = None
        self._load_failed = False
        self._ffmpeg = shutil.which('ffmpeg')
        if self.enabled and self._ffmpeg is None:
            # Without this, a missing ffmpeg makes the whole feature vanish
            # with zero signal anywhere -- no log line, no preview.json, no
            # error. Disable up front (like a failed decoder load) instead of
            # silently no-op-ing every step in _write.
            self.enabled = False
            self.print_fn(
                " - Sample preview disabled: ffmpeg not found on PATH"
            )

    # -- decoder ---------------------------------------------------------
    def _ensure_decoder(self, device):
        if self.decoder is not None or self._load_failed:
            return self.decoder
        from toolkit.models.taehv import load_taehv_decoder

        models_path = os.environ.get('MODELS_PATH', '')
        path = os.path.join(models_path, 'vae_approx', TINY_VAE_BY_ARCH[self.arch])
        try:
            self.decoder = load_taehv_decoder(path, device=device, dtype=torch.float16)
        except Exception as e:
            # Once, then stay quiet -- this runs every step.
            self._load_failed = True
            self.enabled = False
            self.print_fn(f" - Sample preview disabled: could not load {path} ({e})")
        return self.decoder

    # -- writing ---------------------------------------------------------
    def write(self, latents, sample_idx, num_samples, step, total):
        """latents: (N, T, C, H, W) video latents -- the x0 estimate, not the noisy state."""
        if not self.enabled:
            return
        try:
            self._write(latents, sample_idx, num_samples, step, total)
        except Exception as e:
            self.print_fn(f" - Sample preview failed, disabling for this run: {e}")
            self.enabled = False

    def _write(self, latents, sample_idx, num_samples, step, total):
        decoder = self._ensure_decoder(latents.device)
        if decoder is None:
            return

        with torch.no_grad():
            video = decoder(latents[:1].to(decoder.dtype))[0]  # (T, 3, H, W) in [0, 1]

        # Keep the clip's real duration: take every Nth frame rather than the
        # first N, so a 120-frame clip previewed at 60 frames / 12 fps still
        # plays for the five seconds it will actually run for.
        t_total = video.shape[0]
        if 0 < self.frames < t_total:
            idx = np.linspace(0, t_total - 1, self.frames).round().astype(int)
            video = video[torch.from_numpy(idx).to(video.device)]

        frames = video.mul(255).clamp(0, 255).to(torch.uint8)
        frames = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (T, H, W, 3)
        frames = self._downscale(frames)

        out = os.path.join(self.job_folder, 'preview.mp4')
        self._encode(frames, out)

        self._write_json({
            'sample': int(sample_idx) + 1,
            'of': int(num_samples),
            'step': int(step),
            'total': int(total),
            'frames': int(frames.shape[0]),
            'fps': self.fps,
            'updated': time.time(),
            'file': 'preview.mp4',
        })

    def _downscale(self, frames):
        h, w = frames.shape[1:3]
        longest = max(h, w)
        if not self.max_res or longest <= self.max_res:
            return frames
        scale = self.max_res / longest
        from PIL import Image
        new_w, new_h = int(w * scale), int(h * scale)
        return np.stack([
            np.asarray(Image.fromarray(f).resize((new_w, new_h), Image.BILINEAR))
            for f in frames
        ])

    def _encode(self, frames, out_path):
        t, h, w = frames.shape[:3]
        # yuv420p needs even dimensions
        if h % 2 or w % 2:
            frames = frames[:, : h - (h % 2), : w - (w % 2)]
            h, w = frames.shape[1:3]

        tmp = out_path + '.tmp.mp4'
        cmd = [
            self._ffmpeg, '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
            '-framerate', str(self.fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp,
        ]
        # Raw frames over stdin: no temp PNGs, which is most of the encode cost.
        proc = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg: {proc.stderr.decode(errors='replace')[:200]}")
        # Atomic swap so the UI never fetches a half-written file.
        os.replace(tmp, out_path)

    def _write_json(self, payload):
        path = os.path.join(self.job_folder, 'preview.json')
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def cleanup(self):
        """Drop the preview once sampling is over, so a stale clip can't be
        mistaken for a live one."""
        for name in ('preview.json', 'preview.mp4'):
            try:
                os.remove(os.path.join(self.job_folder, name))
            except OSError:
                pass
        self.decoder = None
