"""Tiny video autoencoder decoder (TAEHV), used for live sampling previews.

This is an independent implementation of the TAEHV decoder, derived from the
public architecture and from the shapes in the checkpoint itself. It is
deliberately NOT a copy of ComfyUI's `comfy/taesd/taehv.py`: ComfyUI is GPL-3
and vendoring that file would impose GPL terms on this repository.

Only the decoder is built. Previews decode latents we already have; nothing here
ever needs to encode.

The architecture is pinned by the checkpoint, and `load_taehv_decoder` loads it
with `strict=True` so any mismatch is a hard error at load rather than a subtly
wrong picture later.

Shape contract for `taeh3.safetensors` (MiniMax-H3 video latents):

    input   (N, T, 24, h, w)
    output  (N, T * 4, 3, h * 16, w * 16)   values in [0, 1]

which matches H3's latent geometry: 16x spatial downscale (three 2x upsamples
followed by a 2x pixel shuffle) and 4x temporal (TGrow strides 1, 2, 2).
"""

import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    """Soft clamp on the incoming latent. Sampling can hand us a wild x0 estimate
    in the first step or two, and without this the decode saturates to garbage."""

    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    """Residual block that also sees the previous frame's activations.

    The temporal receptive field of this decoder comes entirely from here: each
    block is convolved over the channel-wise concatenation of the current frame
    and the same block's output for the frame before it, which is why every
    `conv.0` in the checkpoint has twice the block's channel count on its input.
    """

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TGrow(nn.Module):
    """Temporal upsample: widen to `stride` copies of the channels, then unfold
    those copies into new frames. stride 1 is a plain 1x1 projection."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        return self.conv(x)


def apply_with_memory(layers, x, n, t):
    """Run the layer stack over (N*T, C, H, W) input, threading per-frame memory
    through the MemBlocks and growing the time axis at each TGrow.

    Frames are folded into the batch dimension so the 2D convs stay 2D; only the
    MemBlocks and TGrows are aware that time exists.
    """
    for layer in layers:
        if isinstance(layer, MemBlock):
            # (N*T, C, H, W) -> (N, T, C, H, W) so frame i can see frame i-1
            c, h, w = x.shape[-3:]
            xt = x.reshape(n, t, c, h, w)
            # the first frame has no predecessor; it attends to itself, which is
            # what makes a single-frame (image) decode well defined
            past = torch.cat([xt[:, :1], xt[:, :-1]], dim=1)
            x = layer(x, past.reshape(n * t, c, h, w))
        elif isinstance(layer, TGrow):
            x = layer(x)
            stride = layer.stride
            if stride > 1:
                c, h, w = x.shape[-3:]
                # (N, T, C*stride, H, W) -> (N, T*stride, C, H, W): the widened
                # channels ARE the new frames, in order
                x = x.reshape(n, t, stride, c // stride, h, w)
                x = x.reshape(n, t * stride, c // stride, h, w)
                t = t * stride
                x = x.reshape(n * t, c // stride, h, w)
        else:
            x = layer(x)
    return x, t


class TAEHVDecoder(nn.Module):
    """Decoder half of a TAEHV checkpoint.

    Channel widths and block counts are read off the checkpoint rather than
    hardcoded, so this also loads the other taehv-family files (taew2_1,
    taeltx2_3, ...) as long as they share this layout.
    """

    def __init__(self, latent_channels=24, widths=(256, 128, 64), t_strides=(1, 2, 2),
                 blocks_per_stage=3, out_channels=12):
        super().__init__()
        self.latent_channels = latent_channels

        # The final conv emits 3 * shuffle^2 channels; the shuffle factor is the
        # rest of the spatial upscale. It is NOT fixed across the family --
        # taeh3 ends at 12 channels (shuffle 2) while taeltx2_3 ends at 48
        # (shuffle 4) and taew2_1 at 3 (no shuffle) -- so derive it.
        if out_channels % 3 != 0:
            raise ValueError(f"final conv emits {out_channels} channels, not a multiple of 3")
        shuffle_sq = out_channels // 3
        shuffle = int(round(shuffle_sq ** 0.5))
        if shuffle * shuffle != shuffle_sq:
            raise ValueError(f"final conv emits {out_channels} channels; 3 * n^2 expected")
        self.shuffle = shuffle

        self.spatial_upscale = 2 ** len(widths) * shuffle  # 2x per stage, then the shuffle
        self.temporal_upscale = 1
        for s in t_strides:
            self.temporal_upscale *= s

        layers = [Clamp(), conv(latent_channels, widths[0]), nn.ReLU(inplace=True)]
        for stage, width in enumerate(widths):
            layers += [MemBlock(width, width) for _ in range(blocks_per_stage)]
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(TGrow(width, t_strides[stage]))
            next_width = widths[stage + 1] if stage + 1 < len(widths) else widths[-1]
            layers.append(conv(width, next_width, bias=False))
        layers += [nn.ReLU(inplace=True), conv(widths[-1], out_channels)]

        self.layers = nn.ModuleList(layers)
        self.pixel_shuffle = nn.PixelShuffle(shuffle) if shuffle > 1 else nn.Identity()

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, latents):
        """latents: (N, T, C, H, W) -> (N, T * temporal_upscale, 3, H*16, W*16)."""
        n, t = latents.shape[0], latents.shape[1]
        x = latents.reshape(n * t, *latents.shape[2:]).to(self.dtype)
        x, t_out = apply_with_memory(self.layers, x, n, t)
        x = self.pixel_shuffle(x)
        x = x.reshape(n, t_out, *x.shape[1:])
        return x.add_(1).mul_(0.5).clamp_(0, 1)


def _remap_checkpoint(sd):
    """`decoder.<i>....` in the file -> `layers.<i>....` here.

    The checkpoint numbers the decoder as one flat Sequential, and this module
    keeps exactly that numbering so a strict load is a real check on the
    architecture rather than a formality.
    """
    out = {}
    for k, v in sd.items():
        if not k.startswith('decoder.'):
            continue  # encoder half is dead weight for previews
        out['layers.' + k[len('decoder.'):]] = v
    return out


def load_taehv_decoder(path, device='cpu', dtype=torch.float16):
    from safetensors.torch import load_file

    sd = _remap_checkpoint(load_file(path))
    if not sd:
        raise ValueError(f"{path} has no 'decoder.*' tensors -- is it a TAEHV checkpoint?")

    if not all(k.split('.')[1].isdigit() for k in sd):
        # TAESD-family checkpoints (taef1, taesdxl, ...) name their decoder
        # `decoder.layers.N`. Different architecture entirely -- no memblocks, no
        # temporal axis. Fail here rather than half-loading something wrong.
        raise ValueError(f"{path} is not a TAEHV checkpoint (TAESD-style keys); "
                         f"image models need the TAESD decoder instead")

    latent_channels = sd['layers.1.weight'].shape[1]
    last_idx = max(int(k.split('.')[1]) for k in sd)
    out_channels = sd[f'layers.{last_idx}.weight'].shape[0]

    # widths and temporal strides come from the TGrow 1x1 convs, in order
    tgrow = sorted(
        ((int(k.split('.')[1]), v.shape) for k, v in sd.items()
         if k.endswith('.conv.weight') and v.ndim == 4 and v.shape[-1] == 1),
    )
    widths = tuple(shape[1] for _, shape in tgrow)
    t_strides = tuple(shape[0] // shape[1] for _, shape in tgrow)

    model = TAEHVDecoder(
        latent_channels=latent_channels,
        widths=widths,
        t_strides=t_strides,
        out_channels=out_channels,
    )
    model.load_state_dict(sd, strict=True)
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)
