from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
import torch


import random
import time

from pipelines_ootd.pipeline_ootd import OotdPipeline
from pipelines_ootd.unet_garm_2d_condition import UNetGarm2DConditionModel
from pipelines_ootd.unet_vton_2d_condition import UNetVton2DConditionModel
from diffusers import UniPCMultistepScheduler
from diffusers import AutoencoderKL
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer

class OOTDiffusionHDInference:

    def __init__(self,
                 gpu_id,
                 model_root="/workspace/OOTDiffusion/checkpoints/ootd",
                 unet_root="/workspace/OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000",
                 vit_dir="openai/clip-vit-large-patch14",
                 dtype=torch.float16):
        self.gpu_id = 'cuda:' + str(gpu_id)
        vae = AutoencoderKL.from_pretrained(
            model_root,
            subfolder="vae",
            torch_dtype=dtype,
        )
        unet_garm = UNetGarm2DConditionModel.from_pretrained(
            unet_root,
            subfolder="unet_garm",
            torch_dtype=dtype,
            use_safetensors=True,
        )

        unet_vton = UNetVton2DConditionModel.from_pretrained(
            unet_root,
            subfolder="unet_vton",
            torch_dtype=dtype,
            use_safetensors=True,
        )

        self.pipe = OotdPipeline.from_pretrained(
            pretrained_model_name_or_path=model_root,
            unet_garm=unet_garm,
            unet_vton=unet_vton,
            vae=vae,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.gpu_id)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        self.auto_processor = AutoProcessor.from_pretrained(vit_dir)
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(vit_dir).to(self.gpu_id)

        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_root,
            subfolder="tokenizer",
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_root,
            subfolder="text_encoder",
        ).to(self.gpu_id)


    def tokenize_captions(self, captions, max_length):
        inputs = self.tokenizer(
            captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids


    def __call__(self,
                model_type='hd',
                category='upperbody',
                prompt=None,
                image_garm=None,
                image_vton=None,
                mask=None,
                image_ori=None,
                num_samples=1,
                num_steps=20,
                image_scale=1.0,
                seed=-1,
    ):
        if seed == -1:
            random.seed(time.time())
            seed = random.randint(0, 2147483647)
        print('Initial seed: ' + str(seed))
        generator = torch.manual_seed(seed)
        with torch.no_grad():
            prompt_image = self.auto_processor(images=image_garm, return_tensors="pt").to(self.gpu_id)
            batch_size = prompt_image.data['pixel_values'].shape[0]
            prompt_image = self.image_encoder(prompt_image.data['pixel_values']).image_embeds
            prompt_image = prompt_image.unsqueeze(1)
            if model_type == 'hd':
                # Fake caption, note that teh garm's CLIP encoding is
                # also used as a prompt in OOTDiffusion.
                prompt_embeds = self.text_encoder(self.tokenize_captions(prompt, 2).to(self.gpu_id))[0]
                prompt_embeds[:, 1:] = prompt_image[:]
            elif model_type == 'dc':
                prompt_embeds = self.text_encoder(self.tokenize_captions([category] * batch_size, 3).to(self.gpu_id))[0]
                prompt_embeds = torch.cat([prompt_embeds, prompt_image], dim=1)
            else:
                raise ValueError("model_type must be \'hd\' or \'dc\'!")

            images = self.pipe(prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=num_steps,
                        image_guidance_scale=image_scale,
                        num_images_per_prompt=num_samples,
                        generator=generator,
            ).images

        return images