import torch
from toolkit.models.wan21.wan_utils import (
    add_first_frame_conditioning,
    encode_first_frame_condition,
)
from toolkit.prompt_utils import PromptEmbeds
from PIL import Image
import torch
from toolkit.config_modules import GenerateImageConfig
from .wan22_pipeline import Wan22Pipeline

from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO
from diffusers import WanImageToVideoPipeline
from torchvision.transforms import functional as TF

from .wan22_14b_model import Wan2214bModel

class Wan2214bI2VModel(Wan2214bModel):
    arch = "wan22_14b_i2v"
    # get_noise_prediction needs first-frame conditioning every step. For i2v video
    # datasets that is precomputed into the latent cache (see
    # encode_first_frame_condition_for_cache below), so no pixels are needed. Any
    # other dataset shape still falls back to encoding raw pixels per step, which
    # means the raw tensor has to be loaded even when latents are cached.
    requires_pixels_with_cached_latents = True

    def encode_first_frame_condition_for_cache(self, first_frames, latent):
        """Precompute the v2.1 first-frame conditioning latent during latent caching.

        Called by LatentCachingMixin while the VAE is still on the GPU. Deriving
        num_frames and the target size from `latent` the same way
        add_first_frame_conditioning derives them from latent_model_input keeps the
        cached tensor identical to what the per-step path would have produced.
        """
        num_latent_frames = latent.shape[-3]
        num_frames = (num_latent_frames - 1) * 4 + 1
        vae_scale_factor = self.vae.config.scale_factor_spatial
        condition = encode_first_frame_condition(
            first_frame=first_frames,
            num_frames=num_frames,
            target_height=latent.shape[-2] * vae_scale_factor,
            target_width=latent.shape[-1] * vae_scale_factor,
            batch_size=first_frames.shape[0],
            vae=self.vae,
            device=self.device_torch,
            dtype=self.torch_dtype,
        )
        # stored per-item, so drop the batch dim like first_frame_latent does
        return condition.squeeze(0)
    
    
    def generate_single_image(
        self,
        pipeline: Wan22Pipeline,
        gen_config: GenerateImageConfig,
        conditional_embeds: PromptEmbeds,
        unconditional_embeds: PromptEmbeds,
        generator: torch.Generator,
        extra: dict,
    ):
        
        # todo 
        # reactivate progress bar since this is slooooow
        pipeline.set_progress_bar_config(disable=False)

        num_frames = (
            (gen_config.num_frames - 1) // 4
        ) * 4 + 1  # make sure it is divisible by 4 + 1
        gen_config.num_frames = num_frames

        height = gen_config.height
        width = gen_config.width
        first_frame_n1p1 = None
        if gen_config.ctrl_img is not None:
            control_img = Image.open(gen_config.ctrl_img).convert("RGB")

            d = self.get_bucket_divisibility()

            # make sure they are divisible by d
            height = height // d * d
            width = width // d * d

            # resize the control image
            control_img = control_img.resize((width, height), Image.LANCZOS)

            # 5. Prepare latent variables
            # num_channels_latents = self.transformer.config.in_channels
            num_channels_latents = 16
            latents = pipeline.prepare_latents(
                1,
                num_channels_latents,
                height,
                width,
                gen_config.num_frames,
                torch.float32,
                self.device_torch,
                generator,
                None,
            ).to(self.torch_dtype)

            first_frame_n1p1 = (
                TF.to_tensor(control_img)
                .unsqueeze(0)
                .to(self.device_torch, dtype=self.torch_dtype)
                * 2.0
                - 1.0
            )  # normalize to [-1, 1]
            
            # Add conditioning using the standalone function
            gen_config.latents = add_first_frame_conditioning(
                latent_model_input=latents,
                first_frame=first_frame_n1p1,
                vae=self.vae
            )

        def _stop_callback(pipe, i, t, callback_kwargs):
            self.maybe_stop()
            return callback_kwargs

        # keep gen_config in sync with the divisibility-adjusted dims so the
        # LightX2V path (which reads gen_config) matches the direct path
        gen_config.height = height
        gen_config.width = width

        if self._has_lightx2v_loras():
            output = self._lightx2v_generate(
                pipeline,
                gen_config,
                conditional_embeds,
                unconditional_embeds,
                generator,
                extra,
            )
        else:
            output = self._i2v_generate(
                pipeline, gen_config, conditional_embeds, unconditional_embeds,
                generator, extra, height, width, _stop_callback,
            )

        # shape = [1, frames, channels, height, width]
        batch_item = output[0]  # list of pil images
        if gen_config.num_frames > 1:
            return batch_item  # return the frames.
        else:
            # get just the first image
            img = batch_item[0]
        return img

    def _i2v_generate(
        self,
        pipeline,
        gen_config,
        conditional_embeds,
        unconditional_embeds,
        generator,
        extra,
        height,
        width,
        _stop_callback,
    ):
        return pipeline(
            prompt_embeds=conditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            negative_prompt_embeds=unconditional_embeds.text_embeds.to(
                self.device_torch, dtype=self.torch_dtype
            ),
            height=height,
            width=width,
            num_inference_steps=gen_config.num_inference_steps,
            guidance_scale=gen_config.guidance_scale,
            latents=gen_config.latents,
            num_frames=gen_config.num_frames,
            generator=generator,
            return_dict=False,
            output_type="pil",
            callback_on_step_end=_stop_callback,
            **extra,
        )[0]

    def get_noise_prediction(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,  # 0 to 1000 scale
        text_embeddings: PromptEmbeds,
        batch: DataLoaderBatchDTO,
        **kwargs
    ):
        # videos come in (bs, num_frames, channels, height, width)
        # images come in (bs, channels, height, width)
        with torch.no_grad():
            cached_condition = getattr(batch, 'first_frame_conditions', None)
            if cached_condition is not None:
                # Precomputed during latent caching. Nothing below needs a VAE
                # forward pass, so the VAE can stay parked on CPU for the whole run.
                conditioned_latent = add_first_frame_conditioning(
                    latent_model_input=latent_model_input,
                    vae=self.vae,
                    latent_condition=cached_condition,
                )
            else:
                frames = batch.tensor
                if frames is None:
                    raise ValueError(
                        "wan22_14b_i2v needs either a cached first-frame conditioning "
                        "latent or raw pixels. Delete this dataset's _latent_cache so it "
                        "regenerates with 'first_frame_condition', or turn off "
                        "cache_latents_to_disk."
                    )
                if len(frames.shape) == 4:
                    first_frames = frames
                elif len(frames.shape) == 5:
                    first_frames = frames[:, 0]
                else:
                    raise ValueError(f"Unknown frame shape {frames.shape}")

                # the vae may be parked on cpu when latents are cached to disk,
                # but this path needs it to encode the first-frame conditioning
                if self.vae.device != latent_model_input.device:
                    self.vae.to(latent_model_input.device)

                # Add conditioning using the standalone function
                conditioned_latent = add_first_frame_conditioning(
                    latent_model_input=latent_model_input,
                    first_frame=first_frames,
                    vae=self.vae
                )
        
        noise_pred = self.model(
            hidden_states=conditioned_latent,
            timestep=timestep,
            encoder_hidden_states=text_embeddings.text_embeds,
            return_dict=False,
            **kwargs
        )[0]
        return noise_pred