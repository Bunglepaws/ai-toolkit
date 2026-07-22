import json
import os
from typing import List, Optional
import huggingface_hub
import torch
from safetensors.torch import load_file, save_file
from extensions_built_in.audio_models.base_audio_model import BaseAudioModel
from toolkit.basic import flush
from toolkit.config_modules import GenerateImageConfig
from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import get_qtype, quantize, quantize_model

from optimum.quanto import freeze
from .src.model import (
    AceStep15,
    OobleckVAE,
    TextEncoder,
    get_silence_latent,
    infer_dit_config,
    load_models,
)
from transformers import AutoTokenizer
from .src.pipeline import AceStep15Pipeline

scheduler_config = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": False,
}

def to_number(str_or_number, default):
    if isinstance(str_or_number, (int, float)):
        return str_or_number
    if str_or_number is None:
        return default
    if str_or_number == "":
        return default
    try:
        return float(str_or_number)
    except ValueError:
        try:
            return int(str_or_number)
        except ValueError as e:
            raise ValueError(f"Could not convert {str_or_number} to a number") from e


def parse_ace_step_caption(text):
    """Parse a tagged caption file back into a dict."""
    import re

    def tag(name):
        m = re.search(rf"<{name}>(.*?)</{name}>", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    return {
        "caption": tag("CAPTION"),
        "lyrics": tag("LYRICS"),
        "bpm": to_number(tag("BPM"), 120),
        "keyscale": tag("KEYSCALE"),
        "timesignature": tag("TIMESIGNATURE"),
        "duration": to_number(tag("DURATION"), 1.0),
        "language": tag("LANGUAGE"),
    }


class AceStep15Model(BaseAudioModel):
    arch = "ace_step_15"
    sample_rate = 48000

    def __init__(
        self,
        device,
        model_config,
        dtype="bf16",
        custom_pipeline=None,
        noise_scheduler=None,
        **kwargs,
    ):
        super().__init__(
            device, model_config, dtype, custom_pipeline, noise_scheduler, **kwargs
        )
        self.is_flow_matching = True
        self.is_transformer = True
        # self.target_lora_modules = ['AceStep15']
        self.target_lora_modules = ["DiTModel"]
        self.turbo_model = None
        self.audio_lm = None         # Qwen3 LM for audio code generation (lazy-loaded)
        self.audio_quantizer = None  # ResidualFSQ (loaded from AIO on load_model)
        self.audio_detokenizer = None  # AudioTokenDetokenizer (loaded from AIO on load_model)

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def load_model(self):
        # Reset lazy-loaded models so a reused instance doesn't carry over state from a previous job
        self.turbo_model = None
        self.audio_lm = None
        self.audio_quantizer = None
        self.audio_detokenizer = None

        dtype = self.torch_dtype
        device = self.device_torch

        model_path = self.model_config.name_or_path

        if not os.path.exists(model_path):
            # assume it is a hf repo like org/repo/filename.safetensors
            path_parts = model_path.split("/")
            if len(path_parts) != 3:
                raise ValueError(
                    f"Model path {model_path} does not exist and is not a valid Hugging Face repo path"
                )
            model_path = huggingface_hub.hf_hub_download(
                repo_id=f"{path_parts[0]}/{path_parts[1]}",
                filename=path_parts[2],
            )
        # load the models from the single safetensors file
        load_device = device
        if self.model_config.low_vram:
            load_device = "cpu"
            
        models = load_models(model_path, device=load_device, dtype=dtype)

        self.model = models["model"]
        
        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            # quantize_model(self, self.model.decoder)
            quantize(self.model, weights=get_qtype(self.model_config.qtype))
            freeze(self.model)
            flush()
        
        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            self.model.to("cpu")
            
        
        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            raise NotImplementedError("Layer offloading not yet implemented for AceStep15Model")
        
        self.text_encoder = models["text_encoder"]
        
        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(self.text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(self.text_encoder)
            flush()
        
        self.vae = models["vae"]
        
        # move back to device
        self.model.to(device)
        self.text_encoder.to(device)
        self.vae.to(device)
        self.tokenizer = models["tokenizer"]
        
        self.pipeline = AceStep15Pipeline(
            transformer=self.model,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=self.get_train_scheduler(),
        )
        if self.model_config.low_vram:
            self.pipeline.do_tiled_decoding = True

        # Load FSQ quantizer + detokenizer from AIO file (needed for lm_hints at sample time)
        if self.model_config.audio_lm_path:
            self._load_audio_tokenizer(model_path)

        # turbo model and audio LM are loaded lazily on first sample

    def _load_turbo_dit(self, path: str) -> "AceStep15":
        from safetensors.torch import load_file as sf_load_file
        sd = sf_load_file(path)
        # handle both prefixed (model.diffusion_model.*) and bare (decoder.*/encoder.*) key formats
        if any(k.startswith("model.diffusion_model.") for k in sd):
            dit_sd = {k.removeprefix("model.diffusion_model."): v for k, v in sd.items() if k.startswith("model.diffusion_model.")}
        else:
            dit_sd = dict(sd)
        cfg = infer_dit_config(dit_sd)
        turbo = AceStep15(
            hidden=cfg["hidden"], inter=cfg["inter"], heads=cfg["heads"], kv=cfg["kv"],
            head_dim=cfg["head_dim"], n_dit=cfg["n_dit"], n_lyric=cfg["n_lyric"],
            n_timbre=cfg["n_timbre"], enc_hidden=cfg["enc_hidden"],
            enc_heads=cfg["enc_heads"], enc_kv=cfg["enc_kv"], enc_inter=cfg["enc_inter"],
        )
        missing, unexpected = turbo.load_state_dict(dit_sd, strict=False)
        unexpected = [k for k in unexpected if not k.startswith(("tokenizer.", "detokenizer."))]
        if missing:
            self.print_and_status_update(f"  Turbo DiT missing keys: {len(missing)}")
        if unexpected:
            self.print_and_status_update(f"  Turbo DiT unexpected keys: {len(unexpected)}")
        return turbo.to("cpu").to(self.torch_dtype).eval()

    def _compute_lora_deltas(self, lora_path: str) -> dict:
        """Load a LoRA safetensors and return {decoder_param_key: float32_delta}.
        Keys use lora_A (down, [rank, in]) and lora_B (up, [out, rank]). No alpha
        tensors are stored — alpha equals rank so scale is always 1.0.
        """
        from safetensors.torch import load_file as sf_load_file
        sd = sf_load_file(lora_path)
        deltas = {}
        for key in sd:
            if not key.endswith('.lora_B.weight'):
                continue
            if 'decoder.' not in key:
                continue
            a_key = key.replace('.lora_B.weight', '.lora_A.weight')
            if a_key not in sd:
                continue
            lora_A = sd[a_key].float()  # [rank, in]
            lora_B = sd[key].float()    # [out, rank]
            rank = lora_A.shape[0]
            alpha_key = key.replace('.lora_B.weight', '.alpha')
            alpha = float(sd[alpha_key].item()) if alpha_key in sd else float(rank)
            scale = alpha / rank
            delta = (lora_B @ lora_A) * scale  # [out, in]
            # map diffusion_model.decoder.X.lora_B.weight → decoder.X.weight
            param_key = key.replace('diffusion_model.', '').replace('.lora_B.weight', '.weight')
            deltas[param_key] = delta
        return deltas

    def _load_audio_tokenizer(self, aio_path: str):
        """Extract FSQ quantizer + AudioTokenDetokenizer from the AIO safetensors."""
        from safetensors.torch import load_file as sf_load
        from .src.audio_tokenizer import load_audio_tokenizer_from_aio
        self.print_and_status_update("Loading audio tokenizer/detokenizer from AIO...")
        sd = sf_load(aio_path)
        dit_sd = {
            k.removeprefix("model.diffusion_model."): v
            for k, v in sd.items()
            if k.startswith("model.diffusion_model.")
        }
        del sd
        q, d = load_audio_tokenizer_from_aio(dit_sd, device="cpu", dtype=self.torch_dtype)
        self.audio_quantizer = q
        self.audio_detokenizer = d

    def _load_audio_lm(self):
        """Lazy-load Qwen3 LM from audio_lm_path."""
        from .src.audio_lm import load_qwen3_lm
        path = self.model_config.audio_lm_path
        self.print_and_status_update(f"Loading audio LM from {path}")
        self.audio_lm = load_qwen3_lm(path, device="cpu", dtype=self.torch_dtype)

    @torch.no_grad()
    def _generate_context_latents(self, json_prompt: dict, latent_len: int, seed: int, device, dtype):
        """Generate lm_hints context latents via Qwen3 LM → FSQ quantizer → detokenizer."""
        from .src.audio_lm import build_lm_prompts, generate_audio_codes
        from .src.audio_tokenizer import codes_to_context_latents

        caption = json_prompt.get("caption", "")
        lyrics = json_prompt.get("lyrics", "")
        bpm = json_prompt.get("bpm", 120)
        key = json_prompt.get("keyscale", "C major")
        time_sig = json_prompt.get("timesignature", "4/4")
        duration = json_prompt.get("duration", latent_len / 25.0)

        pos_prompt, neg_prompt = build_lm_prompts(caption, lyrics, bpm, duration, key, time_sig)

        # Move LM to device for generation, return to CPU after
        self.audio_lm.to(device)
        try:
            codes = generate_audio_codes(
                lm=self.audio_lm,
                tokenizer=self.tokenizer,
                pos_prompt=pos_prompt,
                neg_prompt=neg_prompt,
                duration=float(duration),
                seed=seed,
                device=device,
                dtype=dtype,
            )
        finally:
            self.audio_lm.to("cpu")

        self.print_and_status_update(f"  Generated {len(codes)} audio codes")

        # Move quantizer/detokenizer to device for the short conversion step
        self.audio_quantizer.to(device)
        self.audio_detokenizer.to(device)
        try:
            ctx = codes_to_context_latents(
                codes, self.audio_quantizer, self.audio_detokenizer,
                latent_len, device, dtype,
            )
        finally:
            self.audio_quantizer.to("cpu")
            self.audio_detokenizer.to("cpu")

        return ctx

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = prompt
        
        if self.text_encoder.device == torch.device("cpu"):
            self.text_encoder.to(self.device_torch)
        # we need the encoder from the model
        if self.model.encoder.device == torch.device("cpu"):
            self.model.encoder.to(self.device_torch)

        # the prompt should be json as a string. Try to parse it.
        json_prompts = []
        for p in prompts:
            try:
                json_prompts.append(parse_ace_step_caption(p))
            except json.JSONDecodeError:
                raise ValueError(
                    f"Prompt {p} is not a valid JSON string. Prompts must be JSON for this model"
                )

        if self.pipeline.text_encoder.device == torch.device("cpu"):
            self.pipeline.text_encoder.to(self.device_torch)

        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        batch_pe = None
        # TODO not sure this will allow for proper batching

        for json_prompt in json_prompts:
            prompt = json_prompt.get("caption", "")
            lyrics = json_prompt.get("lyrics", "")
            bpm = json_prompt.get("bpm", 120)
            key = json_prompt.get("keyscale", "C")
            time_sig = json_prompt.get("timesignature", "4/4")
            duration = json_prompt.get("duration", 10)
            duration = int(duration) if isinstance(duration, (int, float)) else 10
            language = json_prompt.get("language", "en")

            text_embeddings, text_mask, lyric_embeddings, lyric_mask = (
                self.pipeline.get_text_embedings(
                    prompt, lyrics, bpm, key, time_sig, duration, language
                )
            )
            latent_len = int(duration * self.pipeline.LATENT_RATE)
            # Silence as source latent [1, 64, T] -> [1, T, 64] for DiT
            sil = get_silence_latent(latent_len, device, dtype)  # [1, 64, T]
            src = sil.transpose(1, 2)  # [1, T, 64]
            chunk_masks = torch.ones_like(src)

            # Reference audio (silence)
            ref = sil[:, :, :750].transpose(1, 2)  # [1, 750, 64]
            ref_order = torch.zeros(1, device=device, dtype=torch.long)
            enc_h, enc_m, _ = self.pipeline.transformer.prepare_condition(
                text_embeddings,
                text_mask,
                lyric_embeddings,
                lyric_mask,
                ref,
                ref_order,
                src,
                chunk_masks,
            )

            pe = PromptEmbeds(enc_h, attention_mask=enc_m)
            if batch_pe is None:
                batch_pe = pe
            else:
                batch_pe = concat_prompt_embeds(batch_pe, pe)
        return batch_pe

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["layers"]
    
    def get_generation_pipeline(self):
        return self.pipeline

    def generate_single_audio(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        if gen_config.output_ext not in ['mp3', 'wav']:
            gen_config.output_ext = 'mp3'
        prompt = gen_config.prompt
        json_prompt = parse_ace_step_caption(prompt)
        prompt = json_prompt.get("caption", "")
        lyrics = json_prompt.get("lyrics", "")
        bpm = json_prompt.get("bpm", 120)
        key = json_prompt.get("keyscale", "C")
        time_sig = json_prompt.get("timesignature", "4/4")
        duration = json_prompt.get("duration", 0)
        language = json_prompt.get("language", "en")

        if self.model.device == torch.device("cpu"):
            self.model.to(self.device_torch)

        # lazy-load turbo model on first sample (after latent/text caching is done)
        if self.turbo_model is None and self.model_config.turbo_model_path:
            self.print_and_status_update(f"Loading turbo DiT from {self.model_config.turbo_model_path}")
            self.turbo_model = self._load_turbo_dit(self.model_config.turbo_model_path)

        # lazy-load audio LM for lm_hints context generation
        if self.audio_lm is None and self.model_config.audio_lm_path:
            self._load_audio_lm()

        # Generate lm_hints context BEFORE turbo swap so only one large model is on GPU at a time.
        # LM goes to device, generates codes, returns to CPU. Then turbo (if any) goes to device.
        latent_len = int(duration * self.pipeline.LATENT_RATE)
        lm_context = None
        if self.audio_lm is not None and self.audio_quantizer is not None:
            seed = generator.initial_seed() if generator is not None else 0
            lm_context = self._generate_context_latents(
                json_prompt, latent_len, seed, self.device_torch, self.torch_dtype
            )

        using_turbo = self.turbo_model is not None
        turbo_snapshot = {}

        if using_turbo:
            # Merge LoRA into turbo model by loading from the latest saved safetensors.
            # Hook/merge approaches don't work because generate_images calls network.merge_in()
            # before we get here, baking the LoRA into base weights and disabling hooks.
            turbo_lora_path = getattr(self, 'turbo_lora_path', None)
            if turbo_lora_path and os.path.exists(turbo_lora_path):
                self.print_and_status_update(f"Merging LoRA into turbo from {os.path.basename(turbo_lora_path)}")
                lora_deltas = self._compute_lora_deltas(turbo_lora_path)
                turbo_params = dict(self.turbo_model.named_parameters())
                applied = 0
                for param_key, delta in lora_deltas.items():
                    if param_key in turbo_params:
                        p = turbo_params[param_key]
                        # snapshot the pristine weight so we can restore exactly (no bf16 drift)
                        turbo_snapshot[param_key] = p.data.clone()
                        p.data.add_(delta.to(dtype=p.dtype, device=p.device))
                        applied += 1
                self.print_and_status_update(f"  Applied {applied}/{len(lora_deltas)} LoRA deltas to turbo")
            else:
                self.print_and_status_update("Warning: no LoRA checkpoint found for turbo merge — sample will not include LoRA")

            # Use turbo model as pipeline transformer (no hooks — LoRA is baked in above)
            self.model.to("cpu")
            self.turbo_model.to(self.device_torch)
            self.pipeline.transformer = self.turbo_model
            num_steps = 8
            guidance = 1.0
        else:
            num_steps = gen_config.num_inference_steps
            guidance = gen_config.guidance_scale

        try:
            output = self.pipeline(
                prompt=None,
                encoder_embeddings=conditional_embeds.text_embeds.to(self.device_torch, dtype=self.torch_dtype),
                encoder_mask=conditional_embeds.attention_mask.to(self.device_torch, dtype=torch.bool),
                num_inference_steps=num_steps,
                duration=duration,
                generator=generator,
                bpm=bpm,
                key=key,
                time_sig=time_sig,
                language=language,
                guidance_scale=guidance,
                context_latents=lm_context,
            )
        finally:
            if using_turbo:
                # Restore turbo to its pristine weights from the snapshot (exact, no drift)
                turbo_params = dict(self.turbo_model.named_parameters())
                for param_key, original in turbo_snapshot.items():
                    turbo_params[param_key].data.copy_(original.to(turbo_params[param_key].device))
                self.turbo_model.to("cpu")
                self.model.to(self.device_torch)
                self.pipeline.transformer = self.model

        return output

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor, #(1, 300, 64)
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        if self.model.decoder.device == torch.device("cpu"):
            self.model.decoder.to(self.device_torch)
        with torch.no_grad():
            model: AceStep15 = self.model
            tt = timestep.to(self.device_torch, dtype=torch.long) / 1000
            latent_len = latent_model_input.shape[1]
            device = self.device_torch
            dtype = self.torch_dtype
            attn = torch.ones(1, latent_len, device=device, dtype=dtype)

            # build context from silence latent matching the actual input length
            sil = get_silence_latent(latent_len, device, dtype)  # [1, 64, T]
            src = sil.transpose(1, 2)  # [1, T, 64]
            chunk_masks = torch.ones_like(src)
            context = torch.cat([src, chunk_masks], dim=-1)  # [1, T, 128]

        pred = model.decoder(
            x=latent_model_input.detach(),
            timestep=tt.detach(),
            timestep_r=tt.detach(),
            attention_mask=attn.detach(),
            enc_h=text_embeddings.text_embeds.to(self.device_torch, dtype=self.torch_dtype).detach(),
            enc_m=text_embeddings.attention_mask.to(self.device_torch, dtype=torch.bool).detach(),
            context=context.detach(),
        )
        return pred
    
    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()
    
    def encode_audio(self, audio_tensor: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.device_torch
        if dtype is None:
            dtype = self.torch_dtype
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        # .contiguous() ensures a freshly-aligned allocation before the bf16 VAE kernel.
        # Without it, sliced/resampled tensors can land at misaligned addresses on Blackwell.
        audio_in = audio_tensor.to(device=device, dtype=dtype).contiguous()
        output = self.vae.encode(audio_in)
        # transpose from [B, 64, T] to [B, T, 64] for DiT
        output = output.transpose(1, 2).contiguous()
        return output


class AceStep15XLModel(AceStep15Model):
    arch = "ace_step_15_xl"
