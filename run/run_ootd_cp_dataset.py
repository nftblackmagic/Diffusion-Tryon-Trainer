"""Run ootd inference with cp_dataset"""
from pathlib import Path
import sys
import csv

PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from data_scripts.cp_dataset import CPDatasetV2 as CPDataset

import torch
# from preprocess.openpose.run_openpose import OpenPose
# from preprocess.humanparsing.aigc_run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHDInference
from ootd.inference_ootd_dc import OOTDiffusionDC
import argparse

CATEGORY_LIST = ['upperbody', 'lowerbody', 'dress']

def parse_args():
    parser = argparse.ArgumentParser(description='run ootd')
    parser.add_argument('--gpu_id', '-g', type=int, default=0, required=False)
    parser.add_argument('--model_type', type=str, default="hd", required=False)
    parser.add_argument('--category', '-c', type=int, default=0, required=False)
    parser.add_argument('--image_scale', type=float, default=2.0, required=False)
    parser.add_argument('--num_steps', type=int, default=20, required=False)
    parser.add_argument('--num_samples', type=int, default=1, required=False)
    parser.add_argument('--seed', type=int, default=-1, required=False)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument("--test_data_list", type=str, default=None)
    parser.add_argument("--model_root", type=str, default="/workspace/OOTDiffusion/checkpoints/ootd")
    parser.add_argument("--unet_root", type=str, default="/workspace/OOTDiffusion/checkpoints/ootd/ootd_hd/checkpoint-36000")
    parser.add_argument("--vit_dir", type=str, default="openai/clip-vit-large-patch14")
    args = parser.parse_args()
    return args

def build_model(model_type, category):
    # openpose_model = OpenPose(args.gpu_id)
    # parsing_model = Parsing(args.gpu_id)
    if model_type == 'hd' and category != 0:
        raise ValueError("model_type \'hd\' requires category == 0 (upperbody)!")

    if model_type == "hd":
        model = OOTDiffusionHDInference(
            args.gpu_id,
            model_root=args.model_root,
            unet_root=args.unet_root,
            vit_dir=args.vit_dir,
            dtype=torch.float16)
    elif model_type == "dc":
        model = OOTDiffusionDC(args.gpu_id)
    else:
        raise ValueError("model_type must be \'hd\' or \'dc\'!")
    return model

def build_dataloader(args):
    test_dataset = CPDataset(
                    args.dataroot,
                    args.resolution,
                    mode="test",
                    data_list=args.test_data_list)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        # collate_fn=collate_fn_cp,
        batch_size=args.batch_size,
        num_workers=0,
    )
    return test_dataloader

def inference_per_batch(batch, model, args):
    image_garm = batch["ref_imgs"]
    image_vton = batch["inpaint_image"]
    image_ori = batch["GT"]
    # inpaint_mask = batch["inpaint_mask"]
    mask = batch["inpaint_mask"]
    prompt = batch["prompt"]
    file_name = batch["file_name"]
        
    gen_images = model(
        model_type=args.model_type,
        category=CATEGORY_LIST[args.category],
        image_garm=image_garm,
        image_vton=image_vton,
        prompt=prompt,
        mask=mask,
        image_ori=image_ori,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        image_scale=args.image_scale,
        seed=args.seed,
    )
    return gen_images, file_name

def main(args):
    batch_size = args.batch_size
    model_type = args.model_type
    # Build model.
    model = build_model(args.model_type, args.category)
    # Build dataloader
    test_dataloader = build_dataloader(args)

    # Run inference.
    gen_gt_name_pairs = []
    with torch.no_grad():
        for _, batch in enumerate(test_dataloader):
            gen_images, file_name = inference_per_batch(batch, model, args)
            # Save generated images and pred-GT file name pairs for eval。
            for image_idx in range(args.num_samples):
                tmp_images = gen_images[image_idx * batch_size:(image_idx+1) * batch_size]
                for _name, _image in zip(file_name, tmp_images):
                    _gen_img_name = f"out_{_name}_{model_type}_{image_idx}.png"
                    gen_gt_name_pairs.append([_gen_img_name, _name])
                    _image.save(f"./images_output/{_gen_img_name}")
        
    # Write all pairs for eval
    output_file = './test_pairs_for_eval.txt'
    # Writing the data to a txt file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerows(gen_gt_name_pairs)

if __name__ == "__main__":
    args = parse_args()
    main(args)

