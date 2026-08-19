"""Exercise the whole voice-item dataloader path with a stubbed model.

No GPU, no diffusion model, no TTS. Covers everything between "wav on disk" and "batch
ready for get_noise_prediction": scanning, grid snapping, bucketing, the VAE bypass, the
latent cache round trip, and batch collation.
"""
import os, shutil, sys, tempfile, traceback

sys.path.insert(0, r'C:\Data\git\AIToolkitWSL\ai-toolkit')
import torch

from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import AiToolkitDataset
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from toolkit.util.get_model import get_model_class

REAL = r'C:\Data\AIToolkit-StagingArea\datasets\brock_voice'
GRID = get_model_class(type('A', (), {'arch': 'minimax_h3'})()).get_audio_grid()


from toolkit.models.base_model import BaseModel


def _base_flags():
    """Every attribute BaseModel.__init__ sets, without running the heavy __init__
    (device/model loading). Grabbed via __new__ + the flags-only portion of __init__ by
    running __init__ against a throwaway object and reading back __dict__ -- cheaper than
    hand-maintaining a duplicate list that drifts from base_model.py."""
    class _Dummy(BaseModel):
        arch = 'minimax_h3'
        def load_model(self): pass
        def get_generation_pipeline(self): pass
        def generate_single_image(self, *a, **kw): pass
        def get_noise_prediction(self, *a, **kw): pass
        def get_prompt_embeds(self, *a, **kw): pass
        def get_model_has_grad(self): return False
        def get_te_has_grad(self): return False
    from toolkit.config_modules import ModelConfig
    mc = ModelConfig(arch='minimax_h3', name_or_path='dummy', quantize=False)
    d = _Dummy.__new__(_Dummy)
    BaseModel.__init__(d, device='cpu', model_config=mc, dtype='fp32')
    flags = dict(d.__dict__)
    # legacy StableDiffusion-only attributes data_loader.py still checks; False on
    # every non-legacy (BaseModel-based) arch, H3 included
    for k in ('is_xl', 'is_v3', 'is_vega', 'is_ssd', 'is_flux', 'is_auraflow'):
        flags[k] = False
    flags['sample_rate'] = 48000
    return flags


class StubSD:
    """Real BaseModel flag surface (via _base_flags), plus the handful of methods the
    dataloader calls directly."""
    is_audio_model = False

    def get_bucket_divisibility(self):
        return 32
    supports_audio_only_items = True

    class _MC:
        latent_space_version = 'minimax_h3_v1'
    model_config = _MC()
    cache_latents_as_uint8 = False
    encode_control_in_text_embeddings = False
    text_embedding_space_version = 'minimax_h3_v1'
    te_padding_side = 'right'
    latent_space_version = 'minimax_h3_v1'
    torch_dtype = torch.float32
    device_torch = torch.device('cpu')
    arch = 'minimax_h3'
    vae = None
    unet = None
    encode_images_calls = 0

    @classmethod
    def get_audio_grid(cls):
        return GRID

    def make_audio_only_placeholder_latent(self, num_frames, height, width):
        real = get_model_class(type('A', (), {'arch': 'minimax_h3'})())
        return real.make_audio_only_placeholder_latent(self, num_frames, height, width)

    def encode_images(self, imgs):
        StubSD.encode_images_calls += 1
        raise AssertionError('encode_images must NEVER be called for a voice item')

    def encode_audio(self, audio_data_list):
        # mirrors the real packing: (B, 2*T, 32)
        wav = audio_data_list[0]['waveform']
        t = wav.shape[-1] // GRID.hop_length
        return torch.zeros(1, 2 * t, 32)

    def get_frame_count_snapper(self):
        return None

    def set_device_state_preset(self, preset):
        pass

    def restore_device_state(self):
        pass


def check(label, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}{(' -- ' + detail) if detail else ''}")
    if not cond:
        check.failed += 1
check.failed = 0


def main():
    work = tempfile.mkdtemp(prefix='voicetest_')
    ds_dir = os.path.join(work, 'voice')
    shutil.copytree(REAL, ds_dir)
    # a stray hand-cut clip at a NON-grid length, to prove snapping works
    import torchaudio
    torchaudio.save(os.path.join(ds_dir, 'handcut.wav'), torch.zeros(2, 24000 * 4), 24000)

    cfg = DatasetConfig(
        folder_path=ds_dir,
        do_audio=True,
        cache_latents_to_disk=True,
        num_frames=1,               # the case that broke: a voice dataset is NOT "video"
        resolution=256,
        caption_ext='txt',
    )
    sd = StubSD()
    sd.__dict__.update(_base_flags())
    sd.supports_audio_only_items = True
    ds = AiToolkitDataset(dataset_config=cfg, batch_size=1, sd=sd)

    print('\n--- scan ---')
    check('found the clips', len(ds.file_list) == 13, f'{len(ds.file_list)} items')
    voice = [f for f in ds.file_list if getattr(f, 'is_audio_only', False)]
    check('all flagged is_audio_only', len(voice) == 13, f'{len(voice)}')
    check('loss_multiplier zeroed', all(f.loss_multiplier == 0.0 for f in voice))
    check('placeholder canvas 32x32', all(f.width == 32 and f.height == 32 for f in voice))
    frames = sorted({f.num_frames for f in voice})
    check('frame counts on the grid', all(n in GRID.frame_counts for n in frames), str(frames))
    hand = [f for f in voice if 'handcut' in f.path][0]
    check('4.0s hand-cut snapped down to 90f/3.750s', hand.num_frames == 90,
          f'got {hand.num_frames}')

    print('\n--- bucketing ---')
    ds.setup_buckets(quiet=True)
    keys = sorted(ds.buckets.keys())
    check('voice items bucket by duration', len(keys) >= 2, str(keys))
    check('no bucket mixes frame counts',
          all(len({ds.file_list[i].num_frames for i in b.file_list_idx}) == 1
              for b in ds.buckets.values()))
    check('canvas never scaled up to dataset resolution',
          all(ds.file_list[i].crop_width == 32 for b in ds.buckets.values() for i in b.file_list_idx))

    print('\n--- latent cache (VAE bypass) ---')
    ds.cache_latents_all_latents()
    check('encode_images never called', StubSD.encode_images_calls == 0)
    from safetensors.torch import load_file
    cached = 0
    for f in voice:
        lp = f.get_latent_path() if hasattr(f, 'get_latent_path') else None
        if lp and os.path.exists(lp):
            sd_t = load_file(lp)
            cached += 1
            if 'handcut' in f.path:
                check('cache has zeros placeholder', bool((sd_t['latent'] == 0).all()))
                check('cache has audio_latent', 'audio_latent' in sd_t)
                check('cache has num_frames', 'num_frames' in sd_t,
                      str(sd_t.get('num_frames')))
                check('audio latents match the model demand',
                      sd_t['audio_latent'].shape[0] // 2 == GRID.latents_for_frames(f.num_frames),
                      f"{sd_t['audio_latent'].shape} vs {GRID.latents_for_frames(f.num_frames)}")
    check('every voice item cached', cached == 13, f'{cached}')

    print('\n--- batch collation ---')
    b = ds.buckets[keys[0]]
    items = [ds._get_single_item(i) for i in b.file_list_idx[:1]]
    batch = DataLoaderBatchDTO(file_items=items)
    check('batch.latents built', batch.latents is not None,
          str(tuple(batch.latents.shape)) if batch.latents is not None else 'None')
    check('batch.tensor is None (no pixels)', batch.tensor is None)
    check('batch.num_frames from the item', batch.num_frames in GRID.frame_counts,
          str(batch.num_frames))
    check('loss_multiplier_list all zero', all(m == 0.0 for m in batch.loss_multiplier_list),
          str(batch.loss_multiplier_list))
    check('audio present', batch.audio_latents is not None or batch.audio_data is not None)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\n{'ALL PASS' if check.failed == 0 else str(check.failed) + ' FAILURE(S)'}")
    return 1 if check.failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
