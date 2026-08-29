import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.models.v2.vae.qwen_image import QwenImageVAE, QwenImageVAEHolderMixin
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import get_accelerator, unwrap_model
import torch.nn.functional as F
from toolkit.memory_management import MemoryManager
# called below when loading a lora onto a quantized model; upstream dropped
# this import with its own load path, but our path still calls it
from toolkit.util.quantize import filter_lora_state_dict_for_quantized_model
from toolkit.metadata import get_meta_for_safetensors
from safetensors.torch import load_file

from diffusers import (
    QwenImagePipeline,
    AutoencoderKLQwenImage,
)
from transformers import Qwen2VLProcessor
from toolkit.models.v2.diffusion_models.qwen_image import QwenImageTransformer2DModel
from toolkit.models.v2.text_encoders.qwen25_vl import Qwen25VLTextEncoder
from tqdm import tqdm
from toolkit.util.qwen_vae_gradient_checkpointing import patch_qwen_vae_gradient_checkpointing

patch_qwen_vae_gradient_checkpointing()

if TYPE_CHECKING:
    from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO

scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 0.9,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": 0.02,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class QwenImageModel(QwenImageVAEHolderMixin, BaseModel):
    arch = "qwen_image"
    _qwen_image_keep_visual = False
    _qwen_pipeline = QwenImagePipeline

    def __init__(
        self,
        device,
        model_config: ModelConfig,
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
        self.target_lora_modules = ["QwenImageTransformer2DModel"]

    # static method to get the noise scheduler
    @staticmethod
    def get_train_scheduler():
        return CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)

    def get_bucket_divisibility(self):
        return 16 * 2  # 16 for the VAE, 2 for patch size

    def load_model(self):
        dtype = self.torch_dtype
        self.print_and_status_update("Loading Qwen Image model")
        model_path = self.model_config.name_or_path
        base_model_path = self.model_config.extras_name_or_path
        model_dtype = dtype

        if base_model_path.endswith(".safetensors"):
            # use the repo for extras
            base_model_path = "Qwen/Qwen-Image"

        self.print_and_status_update("Loading transformer")

        if not model_path.endswith(".safetensors") and os.path.exists(model_path):
            # check if the path is a full checkpoint.
            te_folder_path = os.path.join(model_path, "text_encoder")
            # if we have the te, this folder is a full checkpoint, use it as the base
            if os.path.exists(te_folder_path):
                base_model_path = model_path

        # load + quantize + offload + placement, all driven by model_config
        transformer = QwenImageTransformer2DModel.load(
            model_path, **self.component_load_kwargs("transformer")
        )
        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = Qwen25VLTextEncoder.load_tokenizer(base_model_path, use_fast=False)
        text_encoder = Qwen25VLTextEncoder.load_model(base_model_path, dtype=dtype)

        # remove the visual model as it is not needed for image generation
        # (before quantization, so the dead tower is never quantized)
        self.processor = None
        if not self._qwen_image_keep_visual:
            text_encoder.drop_vision_tower()

        text_encoder.aitk_post_load(**self.component_load_kwargs("te"))

        self.print_and_status_update("Loading VAE")
        vae = QwenImageVAE.load_model(base_model_path, dtype=dtype)

        self.noise_scheduler = QwenImageModel.get_train_scheduler()

        kwargs = {}

        if self._qwen_image_keep_visual:
            self.print_and_status_update("Loading processor")
            try:
                self.print_and_status_update ("model_path: " + model_path)
                self.processor = Qwen2VLProcessor.from_pretrained(
                    model_path, subfolder="processor"
                )
            except OSError:
                self.print_and_status_update ( "base_model_path: " + base_model_path)
                self.processor = Qwen2VLProcessor.from_pretrained(
                    base_model_path, subfolder="processor"
                )
            kwargs["processor"] = self.processor

        self.print_and_status_update("Making pipe Qwen Image")
        pipe: QwenImagePipeline = self._qwen_pipeline(
            scheduler=self.noise_scheduler,
            text_encoder=None,
            tokenizer=tokenizer,
            vae=vae,
            transformer=None,
            **kwargs,
        )
        self.print_and_status_update("Moving pipe to device")
        # for quantization, it works best to do these after making the pipe
        pipe.text_encoder = text_encoder
        pipe.transformer = transformer

        self.print_and_status_update("Preparing Model")

        text_encoder = [pipe.text_encoder]
        tokenizer = [pipe.tokenizer]

        # leave it on cpu for now
        if not self.low_vram:
            self.print_and_status_update(  "Moving model to device")
            pipe.transformer = pipe.transformer.to(self.device_torch)

        flush()
        # low_vram: the text encoder stays on cpu; get_prompt_embeds moves it
        # to the gpu on demand
        if not self.low_vram:
            text_encoder[0].to(self.device_torch)
        text_encoder[0].requires_grad_(False)
        text_encoder[0].eval()
        flush()

        # save it to the model class
        self.print_and_status_update("Saving model to class")
        self.vae = vae
        self.text_encoder = text_encoder  # list of text encoders
        self.tokenizer = tokenizer  # list of tokenizers
        self.model = pipe.transformer
        self.pipeline = pipe
        self.print_and_status_update("Model Loaded")

    def get_generation_pipeline(self):
        scheduler = QwenImageModel.get_train_scheduler()

        pipeline: QwenImagePipeline = QwenImagePipeline(
            scheduler=scheduler,
            text_encoder=unwrap_model(self.text_encoder[0]),
            tokenizer=self.tokenizer[0],
            vae=unwrap_model(self.vae),
            transformer=unwrap_model(self.transformer),
        )

        pipeline = pipeline.to(self.device_torch)

        return pipeline

    def generate_single_image(
        self,
        pipeline: QwenImagePipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        self.model.to(self.device_torch, dtype=self.torch_dtype)
        control_img = None
        if gen_config.ctrl_img is not None:
            raise NotImplementedError(
                "Control image generation is not supported in Qwen Image model... yet"
            )
            control_img = Image.open(gen_config.ctrl_img)
            control_img = control_img.convert("RGB")
            # resize to width and height
            if control_img.size != (gen_config.width, gen_config.height):
                control_img = control_img.resize(
                    (gen_config.width, gen_config.height), Image.BILINEAR
                )
        self.model.to(self.device_torch)

        # flush for low vram if we are doing that
        flush_between_steps = self.model_config.low_vram

        # Fix a bug in diffusers/torch; also check for stop signal between denoising steps
        def callback_on_step_end(pipe, i, t, callback_kwargs):
            self.maybe_stop()
            if flush_between_steps:
                flush()
            latents = callback_kwargs["latents"]

            return {"latents": latents}

        sc = self.get_bucket_divisibility()
        gen_config.width = int(gen_config.width // sc * sc)
        gen_config.height = int(gen_config.height // sc * sc)

        if self.model_config.low_vram:
            pipeline.vae.enable_tiling()

        if getattr(self, '_sampling_lora_ready', False):
            self._lora_move(pipeline.transformer, "sampling_lora", self.device_torch)
        try:
            img = pipeline(
                prompt_embeds=conditional_embeds.text_embeds,
                prompt_embeds_mask=conditional_embeds.attention_mask.to(
                    self.device_torch, dtype=torch.int64
                ),
                negative_prompt_embeds=unconditional_embeds.text_embeds,
                negative_prompt_embeds_mask=unconditional_embeds.attention_mask.to(
                    self.device_torch, dtype=torch.int64
                ),
                height=gen_config.height,
                width=gen_config.width,
                num_inference_steps=gen_config.num_inference_steps,
                true_cfg_scale=gen_config.guidance_scale,
                latents=gen_config.latents,
                generator=generator,
                callback_on_step_end=callback_on_step_end,
                **extra,
            ).images[0]
        finally:
            if getattr(self, '_sampling_lora_ready', False):
                self._lora_move(pipeline.transformer, "sampling_lora", "cpu")

        if self.model_config.low_vram:
            pipeline.vae.disable_tiling()
        return img

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        **kwargs,
    ):
        self.model.to(self.device_torch)
        batch_size, num_channels_latents, height, width = latent_model_input.shape

        ps = self.transformer.config.patch_size

        # pack image tokens
        latent_model_input = latent_model_input.view(
            batch_size, num_channels_latents, height // ps, ps, width // ps, ps
        )
        latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
        latent_model_input = latent_model_input.reshape(
            batch_size, (height // ps) * (width // ps), num_channels_latents * (ps * ps)
        )

        # img_shapes passed to the model
        img_h2, img_w2 = height // ps, width // ps
        img_shapes = [[(1, img_h2, img_w2)]] * batch_size

        enc_hs = text_embeddings.text_embeds.to(self.device_torch, self.torch_dtype)
        prompt_embeds_mask = text_embeddings.attention_mask.to(
            self.device_torch, dtype=torch.int64
        )

        noise_pred = self.transformer(
            hidden_states=latent_model_input.to(
                self.device_torch, self.torch_dtype
            ).detach(),
            timestep=(timestep / 1000).detach(),
            guidance=None,
            encoder_hidden_states=enc_hs.detach(),
            encoder_hidden_states_mask=prompt_embeds_mask.detach(),
            img_shapes=img_shapes,
            return_dict=False,
            **kwargs,
        )[0]

        # unpack
        noise_pred = noise_pred.view(
            batch_size, height // ps, width // ps, num_channels_latents, ps, ps
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels_latents, height, width)
        return noise_pred

    def get_prompt_embeds(self, prompt: str) -> PromptEmbeds:
        if self.pipeline.text_encoder.device != self.device_torch:
            self.pipeline.text_encoder.to(self.device_torch)

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt,
            device=self.device_torch,
            num_images_per_prompt=1,
        )
        # diffusers >=0.37 returns None when all tokens are valid (no padding)
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(
                prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.int64
            )
        pe = PromptEmbeds(prompt_embeds)
        pe.attention_mask = prompt_embeds_mask
        return pe

    @classmethod
    def validate_sample_lora_paths(cls, model_config, *sample_configs):
        """Called at job startup — raises FileNotFoundError if any configured LoRA path is missing."""
        all_configs = [model_config] + list(sample_configs)
        for cfg in all_configs:
            if cfg is None:
                continue
            path = getattr(cfg, 'sample_lora_path', None)
            if path and not os.path.exists(path):
                raise FileNotFoundError(
                    f"Sampling LoRA path not found (check your config before training starts): {path}"
                )

    def _has_sampling_lora(self):
        sc = getattr(self, 'sample_config', None)
        if sc is not None and getattr(sc, 'sample_lora_path', None):
            return True
        return self.model_config.sample_lora_path is not None

    def _lora_move(self, transformer, adapter_name, device):
        """Move LoRA adapter weights for a transformer to the given device."""
        for module in transformer.modules():
            if hasattr(module, 'lora_A') and adapter_name in module.lora_A:
                module.lora_A[adapter_name].to(device)
                module.lora_B[adapter_name].to(device)

    def _prepare_sampling_lora(self, pipeline):
        """Load the sampling LoRA once and park weights on CPU.

        _before_generate_images_loop calls this once before the sample loop.
        Per-sample code moves weights to GPU and back via _lora_move.
        """
        import peft.tuners.lora.model as _peft_lora_model

        sc = getattr(self, 'sample_config', None)
        lora_path = (getattr(sc, 'sample_lora_path', None) if sc else None) or self.model_config.sample_lora_path
        strength = (getattr(sc, 'sample_lora_strength', None) if sc else None) or self.model_config.sample_lora_strength

        if not os.path.exists(lora_path):
            self.print_and_status_update(f"Warning: sampling LoRA not found: {lora_path}")
            return

        # PEFT 0.18.x bug: dispatch_torchao called without required kwarg on quantized weights
        _orig_dispatch_torchao = _peft_lora_model.dispatch_torchao
        _peft_lora_model.dispatch_torchao = lambda *args, **kwargs: None

        try:
            # Clear any stale adapter registration before loading
            if hasattr(pipeline.transformer, 'peft_config') and 'sampling_lora' in pipeline.transformer.peft_config:
                try:
                    del pipeline.transformer.peft_config['sampling_lora']
                except Exception:
                    pass
            for module in pipeline.transformer.modules():
                if hasattr(module, 'delete_adapter'):
                    try:
                        module.delete_adapter('sampling_lora')
                    except Exception:
                        pass

            from safetensors.torch import load_file as _load_safetensors
            lora_state_dict = _load_safetensors(lora_path)
            n_before = len(lora_state_dict)
            lora_state_dict = filter_lora_state_dict_for_quantized_model(
                pipeline.transformer, lora_state_dict
            )
            n_skipped = n_before - len(lora_state_dict)
            if n_skipped:
                self.print_and_status_update(
                    f"Sampling LoRA: skipping {n_skipped} keys for quantized layers to avoid weight corruption"
                )
            self.print_and_status_update(f"Loading sampling LoRA (strength={strength})")
            # Quanto's QModule._load_from_state_dict unconditionally pops weight._data,
            # raising KeyError when loading a LoRA (which has no _data keys). Patch it to
            # skip quantized reconstruction when no keys for that module are in the state dict.
            try:
                from optimum.quanto.nn.qmodule import QModuleMixin
                _orig_qload = QModuleMixin._load_from_state_dict
                def _safe_qload(self, state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
                    if not any(k.startswith(prefix) for k in state_dict):
                        return
                    return _orig_qload(self, state_dict, prefix, local_metadata, strict,
                                       missing_keys, unexpected_keys, error_msgs)
                QModuleMixin._load_from_state_dict = _safe_qload
                _patched_qmodule = True
            except Exception:
                _patched_qmodule = False

            try:
                pipeline.load_lora_weights(lora_state_dict, adapter_name="sampling_lora")
            finally:
                if _patched_qmodule:
                    QModuleMixin._load_from_state_dict = _orig_qload
            pipeline.set_adapters(["sampling_lora"], adapter_weights=[strength])
            # Park on CPU until needed per-sample
            self._lora_move(pipeline.transformer, "sampling_lora", "cpu")
        finally:
            _peft_lora_model.dispatch_torchao = _orig_dispatch_torchao

        self._sampling_lora_ready = True

    def _teardown_sampling_lora(self, pipeline):
        """Remove the sampling LoRA adapter completely after all sampling is done."""
        if hasattr(pipeline.transformer, 'peft_config') and 'sampling_lora' in pipeline.transformer.peft_config:
            try:
                del pipeline.transformer.peft_config['sampling_lora']
            except Exception:
                pass
        for module in pipeline.transformer.modules():
            if hasattr(module, 'delete_adapter'):
                try:
                    module.delete_adapter('sampling_lora')
                except Exception:
                    pass
        self._sampling_lora_ready = False

    def _validate_sample_config(self, image_configs):
        if not self._has_sampling_lora():
            return
        sc = getattr(self, 'sample_config', None)
        path = (getattr(sc, 'sample_lora_path', None) if sc else None) or self.model_config.sample_lora_path
        if path and not os.path.exists(path):
            raise FileNotFoundError(
                f"Sample LoRA not found — aborting sample to avoid useless inference: {path}"
            )

    def _before_generate_images_loop(self, pipeline, image_configs):
        if self._has_sampling_lora():
            self._prepare_sampling_lora(pipeline)

    def _after_generate_images_loop(self, pipeline):
        if getattr(self, '_sampling_lora_ready', False):
            self._teardown_sampling_lora(pipeline)

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        # comfy-format single-file save (diffusers keys ARE the comfy layout
        # for qwen image); prequantized layers keep their quantized storage
        transformer: QwenImageTransformer2DModel = unwrap_model(self.model)
        if not output_path.endswith(".safetensors"):
            output_path += ".safetensors"
        transformer.save_model(
            output_path,
            dtype=save_dtype,
            metadata=get_meta_for_safetensors(meta, name=self.arch),
        )

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "qwen_image"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    lora_keys_use_comfy_prefix = True

