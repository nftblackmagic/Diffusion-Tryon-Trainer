#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor

from data_scripts.cp_dataset import CPDatasetV2 as CPDataset
import wandb

from ootd.train_ootd_hd import OOTDiffusionModel
from ootd.pipelines_ootd.pipeline_ootd import OotdPipeline as OotdPipelineInference

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(model, args, accelerator, weight_dtype, test_dataloder = None, validation_dataloader = None):
    logger.info("Running validation... ")
    
    unet_garm = accelerator.unwrap_model(model.unet_garm)
    unet_vton = accelerator.unwrap_model(model.unet_vton)

    pipeline = OotdPipelineInference.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=model.vae,
        text_encoder=model.text_encoder,
        tokenizer=model.tokenizer,
        unet_garm=unet_garm,
        unet_vton=unet_vton,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    def sample_imgs(data_loader, log_key: str):
        image_logs = []
        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                with torch.autocast("cuda"):
                    prompt = batch["prompt"][0]
                    image_garm = batch["ref_imgs"][0, :]
                    image_vton = batch["inpaint_image"][0, :]
                    image_ori= batch["GT"][0, :]
                    inpaint_mask = batch["inpaint_mask"][0, :]
                    mask = batch["mask"][0, :].unsqueeze(0)

                    # what is this doing?
                    prompt_image = model.auto_processor(images=image_garm, return_tensors="pt").to(accelerator.device)
                    prompt_image = model.image_encoder(prompt_image.data['pixel_values']).image_embeds
                    prompt_image = prompt_image.unsqueeze(1)
                    prompt_embeds = model.text_encoder(model.tokenize_captions([prompt], 2).to(accelerator.device))[0]
                    prompt_embeds[:, 1:] = prompt_image[:]
                    
                    samples = pipeline(
                        prompt_embeds=prompt_embeds,
                        image_garm=image_garm,
                        image_vton=image_vton, 
                        mask=mask,
                        image_ori=image_ori,
                        num_inference_steps=args.inference_steps,
                        generator=generator,
                    ).images[0]

                    image_logs.append({
                        "garment": image_garm, 
                        "model": image_vton, 
                        "orig_img": image_ori, 
                        "samples": samples, 
                        "prompt": prompt,
                        "inpaint mask": inpaint_mask,
                        "mask": mask
                        })

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                formatted_images = []
                for log in image_logs:
                    formatted_images.append(wandb.Image(log["garment"], caption="garment images"))
                    formatted_images.append(wandb.Image(log["model"], caption="masked model images"))
                    formatted_images.append(wandb.Image(log["orig_img"], caption="original images"))
                    formatted_images.append(wandb.Image(log["inpaint mask"], caption="inpaint mask"))
                    formatted_images.append(wandb.Image(log["mask"], caption="mask"))
                    formatted_images.append(wandb.Image(log["samples"], caption=log["prompt"]))
                tracker.log({log_key: formatted_images})
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")
    
    if validation_dataloader is not None:
        sample_imgs(validation_dataloader, "validation_images")
    if test_dataloder is not None:
        sample_imgs(test_dataloder, "test_images")

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    # TODO: what hell is this?
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    """Only used for pushing the model HF hub."""
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""controlnet-{repo_id} 
    These are controlnet weights trained on {base_model} with new type of conditioning.{img_str}"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="hd",
        help="We will have two types of models, half body and full body.",
    )
    
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=-1, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing_garm",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing_vton",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )

    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    
    parser.add_argument(
        "--dataroot",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--train_data_list",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--validation_data_list",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--test_data_list",
        type=str,
        default=None,
    )

    #TODO: How to set up for multiple GPUs?
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--log_grads",
        action="store_true", help="Whether log the gradients of trained parts."
    )

    parser.add_argument(
        "--vit_path",
        type=str,
        default="openai/clip-vit-large-patch14",
    )

    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--unet_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--notes",
        type=str,
        default="",
    )

    parser.add_argument(
        "--tracker_entity",
        type=str,
        default="catchonlabs",
    )

    parser.add_argument(
        "--clip_grad_norm",
        action="store_true",
        help="if clip the gradients' norm by max_grad_norm"
    )
    
    parser.add_argument(
        "--vton_unet_path",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--garm_unet_path",
        type=str,
        default=None,
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def main(args):
    args.notes = "Train from OOTDiffusion model."
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    # TODO: create tokenizer in OOTD model
    if args.tokenizer_name:
        tokenizer_path = args.tokenizer_name
    elif args.pretrained_model_name_or_path:
        tokenizer_path = args.pretrained_model_name_or_path

    if args.vton_unet_path is None:
        vton_unet_path = args.pretrained_model_name_or_path
    else:
        vton_unet_path = args.vton_unet_path
    
    if args.garm_unet_path is None:
        garm_unet_path = args.pretrained_model_name_or_path
    else:
        garm_unet_path = args.garm_unet_path
    
    # Load scheduler and models
    if args.model_type == "hd":
        # TODO: it is better to move all these paths to args or a config file.
        model = OOTDiffusionModel(
            accelerator.device,
            model_path=args.pretrained_model_name_or_path,
            vton_unet_path=f"{vton_unet_path}",
            garm_unet_path=f"{garm_unet_path}",
            vit_path=args.vit_path
            )
    else:
        raise NotImplementedError(f"Model type {args.model_type} not implemented")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.unet_garm.enable_xformers_memory_efficient_attention()
            model.unet_vton.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    # if unwrap_model(controlnet).dtype != torch.float32:
    #     raise ValueError(
    #         f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
    #     )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    #TODO: use args to control trainning para
    # params_to_optimize = (list(model.unet_garm.parameters()) + list(model.unet_vton.parameters()) + list(model.vae.parameters()))
    
    if args.gradient_checkpointing_vton and args.gradient_checkpointing_garm:
        params_to_optimize = list(model.unet_garm.parameters())  + list(model.unet_vton.parameters())
    elif args.gradient_checkpointing_vton:
        params_to_optimize = list(model.unet_vton.parameters())
    elif args.gradient_checkpointing_garm:
        params_to_optimize = list(model.unet_garm.parameters())

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    if args.dataroot is None:
        assert "Please provide correct data root"
    train_dataset = CPDataset(args.dataroot, args.resolution, mode="train", data_list=args.train_data_list)
    validation_dataset = CPDataset(args.dataroot, args.resolution, mode="train", data_list=args.validation_data_list)
    test_dataset = CPDataset(args.dataroot, args.resolution, mode="test", data_list=args.test_data_list)
    
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.image_encoder.requires_grad_(False)
    
    if args.gradient_checkpointing_vton:
        model.unet_vton.train()
        model.unet_vton.enable_gradient_checkpointing()
    else:
        model.unet_vton.requires_grad_(False)

    if args.gradient_checkpointing_garm:
        model.unet_garm.train()
        model.unet_garm.enable_gradient_checkpointing()
    else:
        model.unet_garm.requires_grad_(False)
    
    

    # Prepare everything with our `accelerator`.
    model.unet_garm, model.vae, model.unet_vton, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model.unet_garm, model.vae, model.unet_vton, optimizer, train_dataloader, test_dataloader, lr_scheduler
    )

    # For mixed precision training we cast untrained weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    model.vae.to(accelerator.device, dtype=weight_dtype)
    model.unet_garm.to(accelerator.device)
    model.unet_vton.to(accelerator.device)
    model.text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config, init_kwargs={"wandb": {"entity": args.tracker_entity}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    unet_garm_grad_dict = defaultdict(list)
    unet_vton_grad_dict = defaultdict(list)
    vae_grad_dict = defaultdict(list)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # pre-train validation
    log_validation(model, args, accelerator, weight_dtype, test_dataloader, validation_dataloader)
    
    # training starts!
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # with accelerator.accumulate(model.unet_garm):
            with torch.autocast("cuda"):
                image_garm = batch["ref_imgs"]
                image_vton = batch["inpaint_image"]
                image_ori = batch["GT"]
                inpaint_mask = batch["inpaint_mask"]
                mask = batch["mask"] # mask will not be used in trainning
                prompt = batch["prompt"]
                                
                prompt_image = model.auto_processor(images=image_garm, return_tensors="pt").to(accelerator.device)
                prompt_image = model.image_encoder(prompt_image.data['pixel_values']).image_embeds
                prompt_image = prompt_image.unsqueeze(1)
                prompt_embeds = model.text_encoder(model.tokenize_captions(prompt, 2).to(accelerator.device))[0]

                prompt_embeds[:, 1:] = prompt_image[:]
                
                prompt_embeds = model._encode_prompt(
                    prompt=prompt,
                    device=accelerator.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                    prompt_embeds=prompt_embeds
                )
                
                image_garm = model.image_processor.preprocess(image_garm)
                image_vton = model.image_processor.preprocess(image_vton)
                image_ori = model.image_processor.preprocess(image_ori)
                mask = mask.unsqueeze(dim=1)
                    
                # Convert images to latent space
                garm_latents = model.prepare_garm_latents(
                    image=image_garm,
                    batch_size=args.train_batch_size,
                    num_images_per_prompt=1,
                    dtype=prompt_embeds.dtype,
                    device=accelerator.device,
                    do_classifier_free_guidance=False,
                )
                
                vton_latents, mask_latents, ori_latents = model.prepare_vton_latents(
                    image=image_vton,
                    mask=mask,
                    image_ori=image_ori,
                    batch_size=args.train_batch_size,
                    num_images_per_prompt=1,
                    dtype=prompt_embeds.dtype,
                    device=accelerator.device,
                    do_classifier_free_guidance=False,
                )

                # TODO: why do we need to use sample() instead of mode()
                latents = model.vae.encode(image_ori).latent_dist.sample()
                # latents = vae.encode(image_ori.to(weight_dtype).latent_dist.sample().to(accelerator.device))
                latents = latents * model.vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)

                
                _, spatial_attn_outputs = model.unet_garm(
                    garm_latents,
                    0,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
                
                latent_vton_model_input = torch.cat([noisy_latents, vton_latents], dim=1)
                spatial_attn_inputs = spatial_attn_outputs.copy()
            
                noise_pred = model.unet_vton(
                    latent_vton_model_input,
                    spatial_attn_inputs,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                
                
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                # TODO: Are these latents x0 or xt-1?
                # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
                # extra_step_kwargs = model.prepare_extra_step_kwargs(generator, args.eta)
                # compute the previous noisy sample x_t -> x_t-1
                # latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                
                accelerator.backward(loss)
                # TODO: Do we need to clip gradients?
                if accelerator.sync_gradients:
                    if args.clip_grad_norm:
                        accelerator.clip_grad_norm_(model.unet_garm.parameters(), args.max_grad_norm)
                        # accelerator.clip_grad_norm_(model.unet_vton.parameters(), args.max_grad_norm)
                        # accelerator.clip_grad_norm_(model.vae.parameters(), args.max_grad_norm)  
                
                
                if args.log_grads:
                    if model.unet_garm.training:
                        for name, block in model.unet_garm.module.named_children():
                            grad = torch.tensor(0.0).to(accelerator.device)
                            for p in block.parameters():
                                if p.grad is not None:
                                    grad += p.grad.norm()
                                    # grad += p.grad.abs().max()
                            unet_garm_grad_dict[name+'.grad.norm'] = grad.detach().item()
                        accelerator.log(unet_garm_grad_dict, step=global_step)
                    
                    if model.unet_vton.training:
                        for name, block in model.unet_vton.module.named_children():
                            grad = torch.tensor(0.0).to(accelerator.device)
                            for p in block.parameters():
                                if p.grad is not None:
                                    grad += p.grad.norm()
                            unet_vton_grad_dict[name+'.grad'] = grad.detach().item()
                        accelerator.log(unet_vton_grad_dict, step=global_step)
                    
                    if model.vae.training:
                        for name, block in model.vae.named_children():
                            grad = torch.tensor(0.0).to(accelerator.device)
                            for p in block.parameters():
                                if p.grad is not None:
                                    grad += p.grad.norm()
                            vae_grad_dict[name+'.grad'] = grad.detach().item()
                        accelerator.log(vae_grad_dict, step=global_step)

                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                

                if accelerator.is_main_process:
                    # Save the checkpoint
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                        unet_vton = unwrap_model(model.unet_vton)
                        unet_vton.save_pretrained(f"{args.output_dir}/unet_vton/checkpoint-{global_step}", safe_serialization=True)
                        
                        unet_garm = unwrap_model(model.unet_vton)
                        unet_garm.save_pretrained(f"{args.output_dir}/unet_garm/checkpoint-{global_step}", safe_serialization=True)

                    if global_step % args.validation_steps == 0:
                        log_validation(
                            model,
                            args,
                            accelerator,
                            weight_dtype,
                            test_dataloader,
                            validation_dataloader,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    
    #Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet_vton = unwrap_model(model.unet_vton)
        unet_vton.save_pretrained(args.output_dir+"/unet_vton", safe_serialization=True)
        
        unet_garm = unwrap_model(model.unet_vton)
        unet_garm.save_pretrained(args.output_dir+"/unet_garm", safe_serialization=True)

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                model,
                args,
                accelerator,
                weight_dtype,
                test_dataloader,
                validation_dataloader,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
