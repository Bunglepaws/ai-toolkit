from functools import partial
import os
from typing import Any, Dict, Optional, Union, List
from typing_extensions import Self
import torch
import yaml
from toolkit.accelerator import unwrap_model
from toolkit.basic import flush
from toolkit.models.wan21.wan_utils import add_first_frame_conditioning
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
from diffusers import UniPCMultistepScheduler
import torch
from toolkit.config_modules import GenerateImageConfig, ModelConfig
from toolkit.samplers.custom_flowmatch_sampler import (
    CustomFlowMatchEulerDiscreteScheduler,
)
from toolkit.util.quantize import quantize_model, has_quant_cache, filter_lora_state_dict_for_quantized_model
from .wan22_pipeline import Wan22Pipeline
from diffusers import WanTransformer3DModel

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from torchvision.transforms import functional as TF

from toolkit.models.wan21.wan21 import Wan21
from .wan22_5b_model import (
    scheduler_config,
    time_text_monkeypatch,
)
from toolkit.memory_management import MemoryManager
from safetensors.torch import load_file, save_file


boundary_ratio_t2v = 0.875
boundary_ratio_i2v = 0.9

scheduler_configUniPC = {
    "_class_name": "UniPCMultistepScheduler",
    "_diffusers_version": "0.35.0.dev0",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "final_sigmas_type": "zero",
    "flow_shift": 3.0,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "bh2",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
}


class DualWanTransformer3DModel(torch.nn.Module):
    def __init__(
        self,
        transformer_1: WanTransformer3DModel,
        transformer_2: WanTransformer3DModel,
        torch_dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        boundary_ratio: float = boundary_ratio_t2v,
        low_vram: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_1: WanTransformer3DModel = transformer_1
        self.transformer_2: WanTransformer3DModel = transformer_2
        self.torch_dtype: torch.dtype = torch_dtype
        self.device_torch: torch.device = device
        self.boundary_ratio: float = boundary_ratio
        self.boundary: float = self.boundary_ratio * 1000
        self.low_vram: bool = low_vram
        self._active_transformer_name = "transformer_1"  # default to transformer_1

    @property
    def device(self) -> torch.device:
        return self.device_torch

    @property
    def dtype(self) -> torch.dtype:
        return self.torch_dtype

    @property
    def config(self):
        return self.transformer_1.config

    @property
    def transformer(self) -> WanTransformer3DModel:
        return getattr(self, self._active_transformer_name)

    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing for both transformers.
        """
        self.transformer_1.enable_gradient_checkpointing()
        self.transformer_2.enable_gradient_checkpointing()

    @torch._dynamo.disable
    def _prepare_transformer(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> None:
        """Select and device-swap the active transformer.

        Decorated with @torch._dynamo.disable because QBytesTensor.detach() (called
        internally by .to()) raises under TorchDynamo tracing when the model is
        quantized with optimum-quanto.  The .item() call at the timestep boundary
        check already causes a graph break; keeping the entire swap logic outside
        the compiler avoids the subsequent re-trace failure."""
        with torch.no_grad():
            t_name = "transformer_1" if timestep.float().mean().item() > self.boundary else "transformer_2"

            if t_name != self._active_transformer_name:
                if self.low_vram:
                    getattr(self, self._active_transformer_name).to("cpu")
                    getattr(self, t_name).to(self.device_torch)
                    torch.cuda.empty_cache()
                self._active_transformer_name = t_name

        if self.transformer.device != hidden_states.device:
            if self.low_vram:
                other_tname = (
                    "transformer_1" if self._active_transformer_name == "transformer_2" else "transformer_2"
                )
                getattr(self, other_tname).to("cpu")
            self.transformer.to(hidden_states.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self._prepare_transformer(hidden_states, timestep)

        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
        )

    def to(self, *args, **kwargs) -> Self:
        # do not do to, this will be handled separately
        return self


class Wan2214bModel(Wan21):
    arch = "wan22_14b"
    _wan_generation_scheduler_config = scheduler_configUniPC
    _wan_expand_timesteps = False
    _wan_vae_path = "ai-toolkit/wan2.1-vae"

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
            device=device,
            model_config=model_config,
            dtype=dtype,
            custom_pipeline=custom_pipeline,
            noise_scheduler=noise_scheduler,
            **kwargs,
        )
        # target it so we can target both transformers
        self.target_lora_modules = ["DualWanTransformer3DModel"]
        self._wan_cache = None

        self.is_multistage = True
        # multistage boundaries split the models up when sampling timesteps
        # for wan 2.2 14b. the timesteps are 1000-875 for transformer 1 and 875-0 for transformer 2
        self.multistage_boundaries: List[float] = [0.875, 0.0]

        self.train_high_noise = model_config.model_kwargs.get("train_high_noise", True)
        self.train_low_noise = model_config.model_kwargs.get("train_low_noise", True)

        self.trainable_multistage_boundaries: List[int] = []
        if self.train_high_noise:
            self.trainable_multistage_boundaries.append(0)
        if self.train_low_noise:
            self.trainable_multistage_boundaries.append(1)

        if len(self.trainable_multistage_boundaries) == 0:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
        
        # if we are only training one or the other, the target LoRA modules will be the wan transformer class
        if not self.train_high_noise or not self.train_low_noise:
            self.target_lora_modules = ["WanTransformer3DModel"]

    def get_transformer_block_names(self):
        # This is called from two different places against two different roots:
        #  - quantize_model() is called once per bare transformer (transformer_1 /
        #    transformer_2) during load_model(), before they're combined — the
        #    right answer there is just "blocks" (relative to that transformer).
        #  - block_compile targets self.unet, which by the time training starts
        #    IS the combined DualWanTransformer3DModel wrapper holding
        #    transformer_1/transformer_2, each with their own .blocks.
        # self.model is only swapped to the wrapper once both transformers are
        # loaded and quantized, so checking it here tells us which caller this is.
        model = getattr(self, 'model', None)
        if isinstance(model, DualWanTransformer3DModel):
            block_names = []
            if self.train_high_noise:
                block_names.append("transformer_1.blocks")
            if self.train_low_noise:
                block_names.append("transformer_2.blocks")
            return block_names
        return ["blocks"]

    def get_quantization_exclude_modules(self):
        # the timestep/text conditioning embedders and the final projection feed
        # every downstream modulation; keep them in full precision when quantizing.
        # names are relative to each individual transformer (they quantize separately)
        return ["condition_embedder*", "proj_out*"]

    @property
    def max_step_saves_to_keep_multiplier(self):
        # the cleanup mechanism checks this to see how many saves to keep
        # if we are training a LoRA, we need to set this to 2 so we keep both the high noise and low noise LoRAs at saves to keep
        if (
            self.network is not None
            and self.network.network_config.split_multistage_loras
        ):
            return 2
        return 1

    def load_model(self):
        # load model from patent parent. Wan21 not immediate parent
        # super().load_model()
        super().load_model()

        # we have to split up the model on the pipeline
        self.pipeline.transformer = self.model.transformer_1
        self.pipeline.transformer_2 = self.model.transformer_2

        # patch the condition embedder
        self.model.transformer_1.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_1.condition_embedder
        )
        self.model.transformer_2.condition_embedder.forward = partial(
            time_text_monkeypatch, self.model.transformer_2.condition_embedder
        )

    def get_bucket_divisibility(self):
        # 8x compression  and 2x2 patch size
        return 16

    def _load_wan_transformer_single_file(self, path, config_subfolder, skip_weights=False):
        # ComfyUI style single-file checkpoint. Diffusers has no Wan2.2 entry in its
        # single-file config inference table (it guesses a Wan2.1 config, leaving
        # mismatched params on the meta device), so load the config from the
        # reference repo and convert the checkpoint keys ourselves.
        from accelerate import init_empty_weights
        from diffusers.loaders.single_file_utils import (
            convert_wan_transformer_to_diffusers,
        )

        default_repo = (
            "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
            if "i2v" in self.arch
            else "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
        )
        config_repo = self.model_config.model_kwargs.get("config_repo", default_repo)
        config = WanTransformer3DModel.load_config(
            config_repo, subfolder=config_subfolder
        )

        if skip_weights:
            # Source file is absent but a quantization cache exists. Create the
            # architecture with random weights — quantize_model will immediately
            # overwrite them from the cache without touching the missing file.
            self.print_and_status_update(
                f" - source file not found, will load from quantization cache"
            )
            transformer = WanTransformer3DModel.from_config(config)
            transformer = transformer.to(self.torch_dtype)
            return transformer

        with init_empty_weights():
            transformer = WanTransformer3DModel.from_config(config)

        state_dict = convert_wan_transformer_to_diffusers(load_file(path))
        dtype = self.torch_dtype
        # cast in place, one tensor at a time, so we never hold a second full
        # copy of the 28GB state dict in RAM
        for k in list(state_dict.keys()):
            state_dict[k] = state_dict[k].to(dtype)
        missing, unexpected = transformer.load_state_dict(
            state_dict, strict=False, assign=True
        )
        if len(missing) > 0:
            raise ValueError(
                f"Checkpoint {path} does not match config {config_repo}/{config_subfolder}, "
                f"missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        if len(unexpected) > 0:
            print(f"Ignoring unexpected keys in {path}: {unexpected[:5]}...")
        return transformer

    def load_wan_transformer(self, transformer_path, subfolder=None):
        if self.model_config.split_model_over_gpus:
            raise ValueError(
                "Splitting model over gpus is not supported for Wan2.2 models"
            )

        if (
            self.model_config.assistant_lora_path is not None
            or self.model_config.inference_lora_path is not None
        ):
            raise ValueError(
                "Assistant LoRA is not supported for Wan2.2 models currently"
            )

        if self.model_config.lora_path is not None:
            raise ValueError(
                "Loading LoRA is not supported for Wan2.2 models currently"
            )

        # transformer path can be a directory that ends with /transformer, a hf path,
        # or a single .safetensors file (ComfyUI style) for the high noise stage with
        # model_kwargs.low_noise_path pointing to the low noise stage file.

        transformer_path_1 = transformer_path
        subfolder_1 = subfolder

        transformer_path_2 = transformer_path
        subfolder_2 = subfolder

        is_single_file = transformer_path.endswith(".safetensors")
        if is_single_file:
            transformer_path_2 = self.model_config.model_kwargs.get(
                "low_noise_path", None
            )
            if transformer_path_2 is None:
                raise ValueError(
                    "model_kwargs.low_noise_path must be set when name_or_path is a "
                    ".safetensors file for Wan2.2 14b (high noise file goes in name_or_path)"
                )
            subfolder_2 = None
        elif subfolder_2 is None:
            # we have a local path, replace it with transformer_2 folder
            transformer_path_2 = os.path.join(
                os.path.dirname(transformer_path_1), "transformer_2"
            )
        else:
            # we have a hf path, replace it with transformer_2 subfolder
            subfolder_2 = "transformer_2"

        self.print_and_status_update("Loading transformer 1")
        dtype = self.torch_dtype
        cache_key_extra_1 = f"high-noise|{transformer_path_1}|{subfolder_1}"
        cache_key_extra_2 = f"low-noise|{transformer_path_2}|{subfolder_2}"
        if is_single_file:
            path_1_exists = os.path.exists(transformer_path_1)
            if not path_1_exists:
                can_use_cache = (
                    self.model_config.quantize
                    and has_quant_cache(self, cache_key_extra_1)
                )
                if not can_use_cache:
                    raise FileNotFoundError(
                        f"Transformer 1 file not found and no quantization cache available: {transformer_path_1}"
                    )
            transformer_1 = self._load_wan_transformer_single_file(
                transformer_path_1, "transformer", skip_weights=not path_1_exists
            )
        else:
            transformer_1 = WanTransformer3DModel.from_pretrained(
                transformer_path_1,
                subfolder=subfolder_1,
                torch_dtype=dtype,
            ).to(dtype=dtype)

        flush()

        if self.model_config.low_vram:
            # quantize on the device
            transformer_1.to('cpu', dtype=dtype)
            flush()
        else:
            transformer_1.to(self.device_torch, dtype=dtype)
            flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 1")
            quantize_model(
                self, transformer_1,
                cache_key_extra=cache_key_extra_1,
            )
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 1 to CPU")
            transformer_1.to("cpu")
        else:
            transformer_1.to(self.device_torch)

        self.print_and_status_update("Loading transformer 2")
        dtype = self.torch_dtype
        if is_single_file:
            path_2_exists = os.path.exists(transformer_path_2)
            if not path_2_exists:
                can_use_cache = (
                    self.model_config.quantize
                    and has_quant_cache(self, cache_key_extra_2)
                )
                if not can_use_cache:
                    raise FileNotFoundError(
                        f"Transformer 2 file not found and no quantization cache available: {transformer_path_2}"
                    )
            transformer_2 = self._load_wan_transformer_single_file(
                transformer_path_2, "transformer_2", skip_weights=not path_2_exists
            )
        else:
            transformer_2 = WanTransformer3DModel.from_pretrained(
                transformer_path_2,
                subfolder=subfolder_2,
                torch_dtype=dtype,
            ).to(dtype=dtype)

        flush()

        if self.model_config.low_vram:
            # quantize on the device
            transformer_2.to('cpu', dtype=dtype)
            flush()
        else:
            transformer_2.to(self.device_torch, dtype=dtype)
            flush()

        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is None:
            # todo handle two ARAs
            self.print_and_status_update("Quantizing Transformer 2")
            quantize_model(
                self, transformer_2,
                cache_key_extra=cache_key_extra_2,
            )
            flush()

        if self.model_config.low_vram:
            self.print_and_status_update("Moving transformer 2 to CPU")
            transformer_2.to("cpu")
        else:
            transformer_2.to(self.device_torch)
    
        layer_offloading_transformer = self.model_config.layer_offloading and self.model_config.layer_offloading_transformer_percent > 0
        # make the combined model
        self.print_and_status_update("Creating DualWanTransformer3DModel")
        transformer = DualWanTransformer3DModel(
            transformer_1=transformer_1,
            transformer_2=transformer_2,
            torch_dtype=self.torch_dtype,
            device=self.device_torch,
            boundary_ratio=boundary_ratio_t2v,
            low_vram=self.model_config.low_vram,
        )
        
        if self.model_config.quantize and self.model_config.accuracy_recovery_adapter is not None:
            # apply the accuracy recovery adapter to both transformers
            self.print_and_status_update("Applying Accuracy Recovery Adapter to Transformers")
            quantize_model(self, transformer)
            flush()
            
        
        if layer_offloading_transformer:
            MemoryManager.attach(
                transformer_1,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_1.scale_shift_table] + [block.scale_shift_table for block in transformer_1.blocks]
            )
            MemoryManager.attach(
                transformer_2,
                self.device_torch,
                offload_percent=self.model_config.layer_offloading_transformer_percent,
                ignore_modules=[transformer_2.scale_shift_table] + [block.scale_shift_table for block in transformer_2.blocks]
            )

        return transformer

    def get_generation_pipeline(self):
        # todo unipc got broken in a diffusers update. Use euler for now.
        # scheduler = UniPCMultistepScheduler(**self._wan_generation_scheduler_config)
        scheduler = self.get_train_scheduler()
        pipeline = Wan22Pipeline(
            vae=self.vae,
            transformer=self.model.transformer_1,
            transformer_2=self.model.transformer_2,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            scheduler=scheduler,
            expand_timesteps=self._wan_expand_timesteps,
            device=self.device_torch,
            aggressive_offload=self.model_config.low_vram,
            # todo detect if it is i2v or t2v
            boundary_ratio=boundary_ratio_t2v,
        )

        # pipeline = pipeline.to(self.device_torch)

        return pipeline

    # static method to get the scheduler
    @staticmethod
    def get_train_scheduler():
        scheduler = CustomFlowMatchEulerDiscreteScheduler(**scheduler_config)
        return scheduler

    def get_base_model_version(self):
        return "wan_2.2_14b"

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs,
    ):
        # todo do we need to override this? Adjust timesteps?
        return super().get_noise_prediction(
            latent_model_input=latent_model_input,
            timestep=timestep,
            text_embeddings=text_embeddings,
            batch=batch,
            **kwargs,
        )

    def get_model_has_grad(self):
        return False

    def get_te_has_grad(self):
        return False

    def save_model(self, output_path, meta, save_dtype):
        transformer_combo: DualWanTransformer3DModel = unwrap_model(self.model)
        transformer_combo.transformer_1.save_pretrained(
            save_directory=os.path.join(output_path, "transformer"),
            safe_serialization=True,
        )
        transformer_combo.transformer_2.save_pretrained(
            save_directory=os.path.join(output_path, "transformer_2"),
            safe_serialization=True,
        )

        meta_path = os.path.join(output_path, "aitk_meta.yaml")
        with open(meta_path, "w") as f:
            yaml.dump(meta, f)

    def save_lora(
        self,
        state_dict: Dict[str, torch.Tensor],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        if not self.network.network_config.split_multistage_loras:
            # just save as a combo lora
            save_file(state_dict, output_path, metadata=metadata)
            return

        # we need to build out both dictionaries for high and low noise LoRAs
        high_noise_lora = {}
        low_noise_lora = {}
        
        only_train_high_noise = self.train_high_noise and not self.train_low_noise
        only_train_low_noise = self.train_low_noise and not self.train_high_noise

        for key in state_dict:
            if ".transformer_1." in key or only_train_high_noise:
                # this is a high noise LoRA
                new_key = key.replace(".transformer_1.", ".")
                high_noise_lora[new_key] = state_dict[key]
            elif ".transformer_2." in key or only_train_low_noise:
                # this is a low noise LoRA
                new_key = key.replace(".transformer_2.", ".")
                low_noise_lora[new_key] = state_dict[key]

        # loras have either LORA_MODEL_NAME_000005000.safetensors or LORA_MODEL_NAME.safetensors
        if len(high_noise_lora.keys()) > 0:
            # save the high noise LoRA
            high_noise_lora_path = output_path.replace(
                ".safetensors", "_high_noise.safetensors"
            )
            save_file(high_noise_lora, high_noise_lora_path, metadata=metadata)

        if len(low_noise_lora.keys()) > 0:
            # save the low noise LoRA
            low_noise_lora_path = output_path.replace(
                ".safetensors", "_low_noise.safetensors"
            )
            save_file(low_noise_lora, low_noise_lora_path, metadata=metadata)

    def load_lora(self, file: str):
        # if it doesnt have high_noise or low_noise, it is a combo LoRA
        if (
            "_high_noise.safetensors" not in file
            and "_low_noise.safetensors" not in file
        ):
            # this is a combined LoRA, we dont need to split it up
            sd = load_file(file)
            return sd

        # we may have been passed the high_noise or the low_noise LoRA path, but we need to load both
        high_noise_lora_path = file.replace(
            "_low_noise.safetensors", "_high_noise.safetensors"
        )
        low_noise_lora_path = file.replace(
            "_high_noise.safetensors", "_low_noise.safetensors"
        )

        combined_dict = {}

        if os.path.exists(high_noise_lora_path) and self.train_high_noise:
            # load the high noise LoRA
            high_noise_lora = load_file(high_noise_lora_path)
            for key in high_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_1."
                )
                combined_dict[new_key] = high_noise_lora[key]
        if os.path.exists(low_noise_lora_path) and self.train_low_noise:
            # load the low noise LoRA
            low_noise_lora = load_file(low_noise_lora_path)
            for key in low_noise_lora:
                new_key = key.replace(
                    "diffusion_model.", "diffusion_model.transformer_2."
                )
                combined_dict[new_key] = low_noise_lora[key]
        
        # if we are not training both stages, we wont have transformer designations in the keys
        if not self.train_high_noise or not self.train_low_noise:
            new_dict = {}
            for key in combined_dict:
                if ".transformer_1." in key:
                    new_key = key.replace(".transformer_1.", ".")
                elif ".transformer_2." in key:
                    new_key = key.replace(".transformer_2.", ".")
                else:
                    new_key = key
                new_dict[new_key] = combined_dict[key]
            combined_dict = new_dict

        return combined_dict
    
    @classmethod
    def validate_sample_lora_paths(cls, model_config, *sample_configs):
        """Called at job startup — raises FileNotFoundError if any configured LoRA path is missing."""
        all_configs = [model_config] + list(sample_configs)
        for cfg in all_configs:
            if cfg is None:
                continue
            for attr, label in [
                ('sample_lora_path', 'LightX2V high noise LoRA'),
                ('sample_lora_path_2', 'LightX2V low noise LoRA'),
            ]:
                path = getattr(cfg, attr, None)
                if path and not os.path.exists(path):
                    raise FileNotFoundError(
                        f"{label} path not found (check your config before training starts): {path}"
                    )

    def _has_lightx2v_loras(self):
        sc = getattr(self, 'sample_config', None)
        if sc is not None:
            if getattr(sc, 'sample_lora_path', None) or getattr(sc, 'sample_lora_path_2', None):
                return True
        return (
            self.model_config.sample_lora_path is not None
            or self.model_config.sample_lora_path_2 is not None
        )

    def _lx2v_move_lora(self, transformer, adapter_name, device):
        """Move LoRA adapter weights for one transformer to the given device."""
        for module in transformer.modules():
            if hasattr(module, 'lora_A') and adapter_name in module.lora_A:
                module.lora_A[adapter_name].to(device)
                module.lora_B[adapter_name].to(device)

    def _prepare_lightx2v_loras(self, pipeline):
        """Load both LightX2V LoRAs once at the start of a sampling session.

        Applies adapters to both transformers, then immediately parks the weights
        on CPU. _lightx2v_generate moves them to GPU per-stage and back to CPU
        after each stage, avoiding a disk load on every sample.
        """
        import peft.tuners.lora.model as _peft_lora_model
        from safetensors.torch import load_file as _load_sf
        from optimum.quanto.nn.qmodule import QModuleMixin as _QMM

        # PEFT 0.18.x bug: dispatch_torchao calls TorchaoLoraLinear without the
        # required get_apply_tensor_subclass kwarg. Bypass it so PEFT falls back
        # to the standard Linear LoRA handler for inference-time adapters on
        # quantized weights.
        _orig_dispatch_torchao = _peft_lora_model.dispatch_torchao
        _peft_lora_model.dispatch_torchao = lambda *args, **kwargs: None

        # quanto interaction bug: PEFT's set_peft_model_state_dict calls
        # load_state_dict(assign=True), whose recursion visits EVERY submodule of
        # the transformer — including the quantized QLinear base_layer modules the
        # training LoRA already wraps. For each, quanto's _load_from_state_dict sees
        # weight_qtype set but no "weight" key in its (LoRA-only) slice, so it tries
        # to rebuild a QTensor from absent _data/_scale keys and raises KeyError.
        # Guard it: when neither the plain weight nor the flattened QTensor data is
        # present there is nothing to load for this module, so skip quanto's
        # reconstruction and leave the existing quantized weight untouched.
        #
        # NOTE: this MUST early-return at the QModuleMixin level. The tempting
        # shortcut of returning None from the inner QBytesTensor loader is wrong —
        # quanto then executes `self.weight = nn.Parameter(None)`, which is an empty
        # [0] tensor, silently destroying every quantized weight it visits.
        _orig_qmm_load = _QMM._load_from_state_dict

        def _safe_qmm_load(self, state_dict, prefix, local_metadata, strict,
                           missing_keys, unexpected_keys, error_msgs):
            weight_name = prefix + "weight"
            if (self.weight_qtype is not None
                    and weight_name not in state_dict
                    and (weight_name + "._data") not in state_dict):
                return
            return _orig_qmm_load(self, state_dict, prefix, local_metadata, strict,
                                  missing_keys, unexpected_keys, error_msgs)

        _QMM._load_from_state_dict = _safe_qmm_load

        def _ensure_adapter_absent(model, adapter_name):
            if hasattr(model, 'peft_config') and adapter_name in model.peft_config:
                try:
                    del model.peft_config[adapter_name]
                except Exception:
                    pass
            for module in model.modules():
                if hasattr(module, 'delete_adapter'):
                    try:
                        module.delete_adapter(adapter_name)
                    except Exception:
                        pass

        try:
            sc = getattr(self, 'sample_config', None)
            high_path = (getattr(sc, 'sample_lora_path', None) if sc else None) or self.model_config.sample_lora_path
            low_path = (getattr(sc, 'sample_lora_path_2', None) if sc else None) or self.model_config.sample_lora_path_2
            high_strength = (getattr(sc, 'sample_lora_strength', None) if sc else None) or self.model_config.sample_lora_strength
            low_strength = (getattr(sc, 'sample_lora_strength_2', None) if sc else None) or self.model_config.sample_lora_strength_2

            if high_path is not None:
                if not os.path.exists(high_path):
                    self.print_and_status_update(f"Warning: LightX2V high noise LoRA not found: {high_path}")
                else:
                    self.print_and_status_update(f"Loading LightX2V high noise LoRA (strength={high_strength})")
                    _ensure_adapter_absent(pipeline.transformer, "lightx2v_high")
                    _high_sd = filter_lora_state_dict_for_quantized_model(pipeline.transformer, _load_sf(high_path))
                    pipeline.load_lora_weights(_high_sd, adapter_name="lightx2v_high")
                    pipeline.set_adapters(["lightx2v_high"], adapter_weights=[high_strength])
                    self._lx2v_move_lora(pipeline.transformer, "lightx2v_high", "cpu")

            if low_path is not None and pipeline.transformer_2 is not None:
                if not os.path.exists(low_path):
                    self.print_and_status_update(f"Warning: LightX2V low noise LoRA not found: {low_path}")
                else:
                    self.print_and_status_update(f"Loading LightX2V low noise LoRA (strength={low_strength})")
                    orig_transformer = pipeline.transformer
                    pipeline.transformer = pipeline.transformer_2
                    try:
                        _ensure_adapter_absent(pipeline.transformer_2, "lightx2v_low")
                        _low_sd = filter_lora_state_dict_for_quantized_model(pipeline.transformer_2, _load_sf(low_path))
                        pipeline.load_lora_weights(_low_sd, adapter_name="lightx2v_low")
                        pipeline.set_adapters(["lightx2v_low"], adapter_weights=[low_strength])
                        self._lx2v_move_lora(pipeline.transformer_2, "lightx2v_low", "cpu")
                    finally:
                        pipeline.transformer = orig_transformer
        finally:
            _peft_lora_model.dispatch_torchao = _orig_dispatch_torchao
            _QMM._load_from_state_dict = _orig_qmm_load

        self._lx2v_loras_ready = True

    def _teardown_lightx2v_loras(self, pipeline):
        """Remove LightX2V LoRA adapters completely after all sampling is done."""
        sc = getattr(self, 'sample_config', None)
        high_path = (getattr(sc, 'sample_lora_path', None) if sc else None) or self.model_config.sample_lora_path
        low_path = (getattr(sc, 'sample_lora_path_2', None) if sc else None) or self.model_config.sample_lora_path_2

        def _delete_adapter(model, adapter_name):
            if hasattr(model, 'peft_config') and adapter_name in model.peft_config:
                try:
                    del model.peft_config[adapter_name]
                except Exception:
                    pass
            for module in model.modules():
                if hasattr(module, 'delete_adapter'):
                    try:
                        module.delete_adapter(adapter_name)
                    except Exception:
                        pass

        if high_path is not None:
            _delete_adapter(pipeline.transformer, "lightx2v_high")
        if low_path is not None and pipeline.transformer_2 is not None:
            _delete_adapter(pipeline.transformer_2, "lightx2v_low")

        self._lx2v_loras_ready = False

    def _validate_sample_config(self, image_configs):
        if not self._has_lightx2v_loras():
            return
        sc = getattr(self, 'sample_config', None)
        high_path = (getattr(sc, 'sample_lora_path', None) if sc else None) or self.model_config.sample_lora_path
        low_path = (getattr(sc, 'sample_lora_path_2', None) if sc else None) or self.model_config.sample_lora_path_2
        for label, path in [("LightX2V high noise LoRA", high_path), ("LightX2V low noise LoRA", low_path)]:
            if path and not os.path.exists(path):
                raise FileNotFoundError(
                    f"Sample LoRA not found — aborting sample to avoid useless inference: {label}: {path}"
                )

    def _before_generate_images_loop(self, pipeline, image_configs):
        if self._has_lightx2v_loras():
            self._prepare_lightx2v_loras(pipeline)

    def _after_generate_images_loop(self, pipeline):
        if getattr(self, '_lx2v_loras_ready', False):
            self._teardown_lightx2v_loras(pipeline)

    def generate_single_image(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        has_lightx2v = self._has_lightx2v_loras()

        def _stop_callback(pipe, i, t, callback_kwargs):
            self.maybe_stop()
            return callback_kwargs

        if self.use_vae_tiling:
            pipeline.vae.enable_tiling()

        if not has_lightx2v:
            output = pipeline(
                prompt_embeds=conditional_embeds.text_embeds.to(
                    self.device_torch, dtype=self.torch_dtype),
                negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                    self.device_torch, dtype=self.torch_dtype),
                height=gen_config.height,
                width=gen_config.width,
                num_inference_steps=gen_config.num_inference_steps,
                guidance_scale=gen_config.guidance_scale,
                latents=gen_config.latents,
                num_frames=gen_config.num_frames,
                generator=generator,
                return_dict=False,
                output_type="pil",
                callback_on_step_end=_stop_callback,
                **extra
            )[0]
        else:
            output = self._lightx2v_generate(pipeline, gen_config, conditional_embeds, unconditional_embeds, generator, extra)

        if self.use_vae_tiling:
            # restore no tiling
            pipeline.vae.disable_tiling()

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def _lightx2v_generate(
        self,
        pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        """Two-stage LightX2V distillation inference.

        Stage 1: high-noise model (transformer_1 + high LoRA) for first half of steps.
        Stage 2: low-noise model (transformer_2 + low LoRA) for second half of steps,
                 starting from stage 1's output latent.

        Text encoder is NOT reloaded — pre-computed embeddings from the training cache
        are used directly.
        """
        num_steps = gen_config.num_inference_steps
        mid_step = num_steps // 2

        cond_embeds = conditional_embeds.text_embeds.to(self.device_torch, dtype=self.torch_dtype)
        uncond_embeds = unconditional_embeds.text_embeds.to(self.device_torch, dtype=self.torch_dtype)

        def _stop_callback(pipe, i, t, callback_kwargs):
            self.maybe_stop()
            return callback_kwargs

        common_kwargs = dict(
            prompt_embeds=cond_embeds,
            negative_prompt_embeds=uncond_embeds,
            height=gen_config.height,
            width=gen_config.width,
            num_frames=gen_config.num_frames,
            num_inference_steps=num_steps,
            guidance_scale=gen_config.guidance_scale,
            generator=generator,
            return_dict=False,
            callback_on_step_end=_stop_callback,
            **extra
        )

        # Disable boundary-ratio transformer switching so each stage uses only its own
        # transformer rather than the pipeline routing by timestep threshold.
        orig_boundary_ratio = pipeline.config.boundary_ratio
        pipeline.register_to_config(boundary_ratio=None)

        try:
            # Stage 1: high-noise model (transformer_1) with high LoRA
            self._lx2v_move_lora(pipeline.transformer, "lightx2v_high", self.device_torch)
            try:
                intermediate_latents = pipeline(
                    **common_kwargs,
                    latents=gen_config.latents,
                    output_type="latent",
                    denoising_end_step=mid_step,
                )[0]
            finally:
                self._lx2v_move_lora(pipeline.transformer, "lightx2v_high", "cpu")

            # i2v: the pipeline strips the 20 first-frame conditioning channels and
            # returns bare 16-channel latents, so re-attach the conditioning for stage 2
            if gen_config.latents is not None and gen_config.latents.shape[1] == 36:
                conditioning = gen_config.latents[:, 16:].to(
                    intermediate_latents.device, intermediate_latents.dtype
                )
                intermediate_latents = torch.cat(
                    [intermediate_latents, conditioning], dim=1
                )

            # Stage 2: low-noise model (transformer_2) with low LoRA.
            # Swap pipeline.transformer so the denoising loop uses transformer_2
            # (boundary_ratio is None so it always uses self.transformer).
            orig_transformer = pipeline.transformer  # = transformer_1
            if self.model_config.low_vram:
                orig_transformer.to("cpu")
            pipeline.transformer = pipeline.transformer_2
            self._lx2v_move_lora(pipeline.transformer_2, "lightx2v_low", self.device_torch)
            try:
                output = pipeline(
                    **common_kwargs,
                    latents=intermediate_latents,
                    output_type="pil",
                    denoising_start_step=mid_step,
                )[0]
            finally:
                self._lx2v_move_lora(pipeline.transformer_2, "lightx2v_low", "cpu")
                pipeline.transformer = orig_transformer
                if self.model_config.low_vram:
                    pipeline.transformer.to(self.device_torch)
        finally:
            pipeline.register_to_config(boundary_ratio=orig_boundary_ratio)

        return output

    def get_model_to_train(self):
        # todo, loras wont load right unless they have the transformer_1 or transformer_2 in the key.
        # called when setting up the LoRA. We only need to get the model for the stages we want to train.
        if self.train_high_noise and self.train_low_noise:
            # we are training both stages, return the unified model
            return self.model
        elif self.train_high_noise:
            # we are only training the high noise stage, return transformer_1
            return self.model.transformer_1
        elif self.train_low_noise:
            # we are only training the low noise stage, return transformer_2
            return self.model.transformer_2
        else:
            raise ValueError(
                "At least one of train_high_noise or train_low_noise must be True in model.model_kwargs"
            )
