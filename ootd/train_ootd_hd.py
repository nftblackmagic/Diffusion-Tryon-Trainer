import pdb
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Any, Callable, Dict, List, Optional, Union


import random
import time
import pdb
import PIL.Image


from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers import DDPMScheduler
from diffusers.image_processor import VaeImageProcessor
from diffusers import AutoencoderKL
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import load_file

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class OOTDiffusionModel:

    def __init__(self, device, model_path, **kwargs):
        # self.gpu_id = 'cuda:' + str(gpu_id)
        self.device = device

        MODEL_PATH = model_path
        UNET_PATH = kwargs["unet_path"] if "unet_path" in kwargs else MODEL_PATH
        GARM_UNET_PATH = kwargs["garm_unet_path"] if "garm_unet_path" in kwargs else UNET_PATH
        VTON_UNET_PATH = kwargs["vton_unet_path"] if "vton_unet_path" in kwargs else UNET_PATH
        VIT_PATH = kwargs["vit_path"] if "vit_path" in kwargs else MODEL_PATH
        VAE_PATH = kwargs["vae_path"] if "vae_path" in kwargs else MODEL_PATH

        self.vae = AutoencoderKL.from_pretrained(
            VAE_PATH,
            subfolder="vae",
            torch_dtype=torch.float16,
        )
    
        # unet_sd = load_file(f"{MODEL_PATH}/diffusion_pytorch_model.safetensors")
        self.unet_garm = UNetGarm2DConditionModel.from_pretrained(
            GARM_UNET_PATH,
            subfolder="unet",
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )

        self.unet_vton = UNetVton2DConditionModel.from_pretrained(
            VTON_UNET_PATH,
            subfolder="unet",
            # torch_dtype=torch.float16,
            use_safetensors=True,
            low_cpu_mem_usage=False,
            ignore_mismatched_sizes=True
        )
        
        def replace_first_conv_layer(unet_model, new_in_channels):
            # Access the first convolutional layer
            # This example assumes the first conv layer is directly an attribute of the model
            # Adjust the attribute access based on your model's structure
            original_first_conv = unet_model.conv_in
            
            # Create a new Conv2d layer with the desired number of input channels
            # and the same parameters as the original layer
            new_first_conv = torch.nn.Conv2d(
                in_channels=new_in_channels,
                out_channels=original_first_conv.out_channels,
                kernel_size=original_first_conv.kernel_size,
                padding=1,
            )
            
            # Zero-initialize the weights of the new convolutional layer
            new_first_conv.weight.data.zero_()

            # Copy the bias from the original convolutional layer to the new layer
            new_first_conv.bias.data = original_first_conv.bias.data.clone()
            
            new_first_conv.weight.data[:, :original_first_conv.in_channels] = original_first_conv.weight.data
            
            # Replace the original first conv layer with the new one
            return new_first_conv

        self.unet_vton.conv_in = replace_first_conv_layer(self.unet_vton, 8)  #replace the conv in layer from 4 to 8 to make sd15 match with new input dims
        
        self.auto_processor = AutoProcessor.from_pretrained(VIT_PATH)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(VIT_PATH).to(self.device)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            MODEL_PATH,
            subfolder="tokenizer",
        )

        self.text_encoder = CLIPTextModel.from_pretrained(
            MODEL_PATH,
            subfolder="text_encoder",
        ).to(self.device)

        self.scheduler = DDPMScheduler.from_pretrained(
            MODEL_PATH, 
            subfolder="scheduler"
            )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        
    def tokenize_captions(self, captions, max_length):
        
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        return inputs.input_ids
    
    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds])

        return prompt_embeds

    def prepare_garm_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.mode()

        # if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
        #     additional_image_per_prompt = batch_size // image_latents.shape[0]
        #     image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        # elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
        #     raise ValueError(
        #         f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
        #     )
        # else:
        image_latents = torch.cat([image_latents], dim=0)

        # if do_classifier_free_guidance:
        #     uncond_image_latents = torch.zeros_like(image_latents)
        #     image_latents = torch.cat([image_latents, uncond_image_latents], dim=0)

        return image_latents
    
    def prepare_vton_latents(
        self, image, mask, image_ori, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(
                f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}"
            )

        image = image.to(device=device, dtype=dtype)
        image_ori = image_ori.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
            image_ori_latents = image_ori
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                image_latents = [self.vae.encode(image[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                image_latents = torch.cat(image_latents, dim=0)
                image_ori_latents = [self.vae.encode(image_ori[i : i + 1]).latent_dist.mode() for i in range(batch_size)]
                image_ori_latents = torch.cat(image_ori_latents, dim=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.mode()
                image_ori_latents = self.vae.encode(image_ori).latent_dist.mode()

        mask = torch.nn.functional.interpolate(
            mask, size=(image_latents.size(-2), image_latents.size(-1))
        )
        mask = mask.to(device=device, dtype=dtype)

        image_latents = torch.cat([image_latents], dim=0)
        mask = torch.cat([mask], dim=0)
        image_ori_latents = torch.cat([image_ori_latents], dim=0)


        return image_latents, mask, image_ori_latents

