import os
from typing import TYPE_CHECKING, List, Optional

import torch
import yaml
from toolkit import train_tools
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from PIL import Image
from toolkit.models.base_model import BaseModel
from toolkit.basic import flush
from toolkit.prompt_utils import PromptEmbeds
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.accelerator import get_accelerator, unwrap_model
from optimum.quanto import freeze, QTensor
from toolkit.util.quantize import quantize, get_qtype, quantize_model, filter_lora_state_dict_for_quantized_model
import torch.nn.functional as F
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file

from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2Tokenizer,
    Qwen2VLProcessor,
)
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


class QwenImageModel(BaseModel):
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

        if model_path.endswith(".safetensors"):
            # load the safetensors file
            transformer = QwenImageTransformer2DModel.from_single_file(
                model_path,
                config="Qwen/Qwen-Image",
                subfolder="transformer",
                torch_dtype=model_dtype,
            )
            transformer.to(model_dtype)

        else:
            transformer_path = model_path
            transformer_subfolder = "transformer"
            if os.path.exists(transformer_path):
                transformer_subfolder = None
                transformer_path = os.path.join(transformer_path, "transformer")
                # check if the path is a full checkpoint.
                te_folder_path = os.path.join(model_path, "text_encoder")
                # if we have the te, this folder is a full checkpoint, use it as the base
                if os.path.exists(te_folder_path):
                    base_model_path = model_path

            transformer = QwenImageTransformer2DModel.from_pretrained(
                transformer_path, subfolder=transformer_subfolder, torch_dtype=dtype
            )

        if self.model_config.quantize:
            self.print_and_status_update("Quantizing Transformer")
            quantize_model(self, transformer)
            flush()

        self.maybe_stop()

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_transformer_percent > 0
        ):
            MemoryManager.attach(
                transformer,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
            )

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer to CPU")
            transformer.to("cpu")

        flush()

        self.print_and_status_update("Text Encoder")
        tokenizer = Qwen2Tokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer", torch_dtype=dtype
        )
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path, subfolder="text_encoder", torch_dtype=dtype
        )

        # remove the visual model as it is not needed for image generation
        self.processor = None
        if not self._qwen_image_keep_visual:
            text_encoder.model.visual = None

        if (
            self.model_config.layer_offloading
            and self.model_config.layer_offloading_text_encoder_percent > 0
        ):
            MemoryManager.attach(
                text_encoder,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_text_encoder_percent,
            )

        text_encoder.to(self.device_torch, dtype=dtype)
        flush()

        self.maybe_stop()

        if self.model_config.quantize_te:
            self.print_and_status_update("Quantizing Text Encoder")
            quantize(text_encoder, weights=get_qtype(self.model_config.qtype_te))
            freeze(text_encoder)
            flush()

        self.print_and_status_update("Loading VAE")
        vae = AutoencoderKLQwenImage.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype
        )

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
        # just to make sure everything is on the right device and dtype
        self.print_and_status_update("Moving text encoder to device")
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
        # only save the unet
        transformer: QwenImageTransformer2DModel = unwrap_model(self.model)
        transformer.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def get_loss_target(self, *args, **kwargs):
        noise = kwargs.get("noise")
        batch = kwargs.get("batch")
        return (noise - batch.latents).detach()

    def get_base_model_version(self):
        return "qwen_image"

    def get_transformer_block_names(self) -> Optional[List[str]]:
        return ["transformer_blocks"]

    def convert_lora_weights_before_save(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("transformer.", "diffusion_model.")
            new_sd[new_key] = value
        return new_sd

    def convert_lora_weights_before_load(self, state_dict):
        new_sd = {}
        for key, value in state_dict.items():
            new_key = key.replace("diffusion_model.", "transformer.")
            new_sd[new_key] = value
        return new_sd

    def encode_images(self, image_list: List[torch.Tensor], device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        # Move to vae to device if on cpu
        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        # move to device and dtype
        image_list = [image.to(device, dtype=dtype) for image in image_list]
        images = torch.stack(image_list).to(device, dtype=dtype)
        # it uses wan vae, so add dim for frame count

        images = images.unsqueeze(2)
        latents = self.vae.encode(images).latent_dist.sample()

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)

        latents = (latents - latents_mean) * latents_std
        latents = latents.to(device, dtype=dtype)

        latents = latents.squeeze(2)  # remove the frame count dimension

        return latents

    def decode_latents(self, latents: torch.Tensor, device=None, dtype=None):
        if device is None:
            device = self.vae_device_torch
        if dtype is None:
            dtype = self.vae_torch_dtype

        if self.vae.device == torch.device("cpu"):
            self.vae.to(device)

        latents = latents.to(device, dtype=dtype)

        # add frame count dim for wan vae
        latents = latents.unsqueeze(2)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents = latents * latents_std + latents_mean

        images = self.vae.decode(latents).sample

        images = images.squeeze(2)  # remove the frame count dimension

        return images.to(device, dtype=dtype)
