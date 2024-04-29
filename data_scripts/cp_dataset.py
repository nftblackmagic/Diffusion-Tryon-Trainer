# coding=utf-8
import json
import os
import os.path as osp
import random
from copy import deepcopy

import cv2
import numpy as np
import pandas as pd
import PIL
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader


def mask2bbox(mask):
    up = np.max(np.where(mask)[0])
    down = np.min(np.where(mask)[0])
    left = np.min(np.where(mask)[1])
    right = np.max(np.where(mask)[1])
    center = ((up + down) // 2, (left + right) // 2)

    factor = random.random() * 0.1 + 0.1

    up = int(min(up * (1 + factor) - center[0] * factor + 1, mask.shape[0]))
    down = int(max(down * (1 + factor) - center[0] * factor, 0))
    left = int(max(left * (1 + factor) - center[1] * factor, 0))
    right = int(min(right * (1 + factor) - center[1] * factor + 1, mask.shape[1]))
    return (down, up, left, right)

debug_mode=True

def tensor_to_image(tensor, image_path):
    """
    Convert a torch tensor to an image file.

    Args:
    - tensor (torch.Tensor): the input tensor. Shape (C, H, W).
    - image_path (str): path where the image should be saved.

    Returns:
    - None
    """
    if debug_mode: 
        # Check the tensor dimensions. If it's a batch, take the first image
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        # Check for possible normalization and bring the tensor to 0-1 range if necessary
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

        # Convert tensor to PIL Image
        to_pil = ToPILImage()
        img = to_pil(tensor)

        # Save the PIL Image
        dir_path = os.path.dirname(image_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        img.save(image_path)


class CPDataset(data.Dataset):
    """
    Dataset for CP-VTON.
    """

    def __init__(self, dataroot, image_size=512, mode="train", data_list: str = "train_pairs.txt", semantic_nc=13, unpaired=False):
        super(CPDataset, self).__init__()
        # base setting
        self.root = dataroot
        self.unpaired = unpaired
        self.datamode = mode  # train or test or self-defined
        self.data_list = data_list
        self.fine_height = image_size
        self.fine_width = int(image_size / 256 * 256)
        self.semantic_nc = semantic_nc
        self.data_path = osp.join(dataroot, mode)
        self.crop_size = (self.fine_height, self.fine_width)
        self.toTensor = transforms.ToTensor()
        # self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.Compose([transforms.ToTensor()])
        # self.clip_normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        # load data list
        im_names = []
        c_names = []
        with open(osp.join(dataroot, self.data_list), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

        # load caption dict
        # caption_file_name = data_list.replace(".txt", "_captions.json")
        # with open(osp.join(dataroot, caption_file_name), "r") as f:
        #     self.caption_dict = json.load(f)

    def name(self):
        return "CPDataset"

    def get_agnostic(self, im, im_parse, pose_data):
        parse_array = np.array(im_parse)
        parse_head = (parse_array == 4).astype(np.float32) + (parse_array == 13).astype(np.float32)
        parse_lower = (
            (parse_array == 9).astype(np.float32)
            + (parse_array == 12).astype(np.float32)
            + (parse_array == 16).astype(np.float32)
            + (parse_array == 17).astype(np.float32)
            + (parse_array == 18).astype(np.float32)
            + (parse_array == 19).astype(np.float32)
        )

        agnostic = im.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)

        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        point = (pose_data[9] + pose_data[12]) / 2
        pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
        pose_data[12] = point + (pose_data[12] - point) / length_b * length_a

        r = int(length_a / 16) + 1

        # mask torso
        for i in [9, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), "gray", "gray")
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], "gray", width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], "gray", width=r * 6)
        agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], "gray", width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], "gray", "gray")

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle((pointx - r * 5, pointy - r * 9, pointx + r * 5, pointy), "gray", "gray")

        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], "gray", width=r * 12)
        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), "gray", "gray")
        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], "gray", width=r * 10)
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), "gray", "gray")

        for parse_id, pose_ids in [(14, [5, 6, 7]), (15, [2, 3, 4])]:
            # mask_arm = Image.new('L', (self.fine_width, self.fine_height), 'white')
            mask_arm = Image.new("L", (768, 1024), "white")
            mask_arm_draw = ImageDraw.Draw(mask_arm)
            pointx, pointy = pose_data[pose_ids[0]]
            mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 6, pointx + r * 5, pointy + r * 6), "black", "black")
            for i in pose_ids[1:]:
                if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                    continue
                mask_arm_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], "black", width=r * 10)
                pointx, pointy = pose_data[i]
                if i != pose_ids[-1]:
                    mask_arm_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), "black", "black")
            mask_arm_draw.ellipse((pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4), "black", "black")

            parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
            agnostic.paste(im, None, Image.fromarray(np.uint8(parse_arm * 255), "L"))

        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_head * 255), "L"))
        agnostic.paste(im, None, Image.fromarray(np.uint8(parse_lower * 255), "L"))
        return agnostic

    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = "image/" + im_name
        c_name = {}
        c = {}
        cm = {}
        if self.unpaired:
            key = "unpaired"
        else:
            key = "paired"

        c_name[key] = self.c_names[key][index]
        c[key] = Image.open(osp.join(self.data_path, "cloth", c_name[key])).convert("RGB")
        c[key] = transforms.Resize(self.crop_size, interpolation=2)(c[key])
        c_img = c[key]
        ref_image_pa = self.transform(c_img)
        cm[key] = Image.open(osp.join(self.data_path, "cloth-mask", c_name[key]))
        cm[key] = transforms.Resize(self.crop_size, interpolation=0)(cm[key])
        cm_img = cm[key]

        c[key] = self.transform(c[key])  # [-1,1]
        cm_array = np.array(cm[key])
        cm_array = (cm_array >= 128).astype(np.float32)
        cm[key] = torch.from_numpy(cm_array)  # [0,1]
        cm[key].unsqueeze_(0)

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize(self.crop_size, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)
        # TODO: save these to wandb
        tensor_to_image(im, "./internal/im.jpg")

        # load parsing image
        parse_name = im_name.replace("image", "image-parse-v3").replace(".jpg", ".png")
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize(self.crop_size, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

        # parse map
        labels = {
            0: ["background", [0, 10]],
            1: ["hair", [1, 2]],
            2: ["face", [4, 13]],
            3: ["upper", [5, 6, 7]],
            4: ["bottom", [9, 12]],
            5: ["left_arm", [14]],
            6: ["right_arm", [15]],
            7: ["left_leg", [16]],
            8: ["right_leg", [17]],
            9: ["left_shoe", [18]],
            10: ["right_shoe", [19]],
            11: ["socks", [8]],
            12: ["noise", [3, 11]],
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        mask_id = torch.Tensor([3, 5, 6])
        mask = torch.isin(parse_onehot[0], mask_id).numpy()

        kernel_size = int(5 * (self.fine_width / 256))
        mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=3)
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=1)
        mask = mask.astype(np.float32)
        # TODO: to work with OOTDpipeline, which mask do we need?
        inpaint_mask = 1 - self.toTensor(mask)
        # inpaint_mask = self.toTensor(mask)

        warped_cloth_name = im_name.replace("image", "cloth-warp" if not self.unpaired else "unpaired-cloth-warp")

        warped_cloth = Image.open(osp.join(self.data_path, warped_cloth_name))
        warped_cloth = transforms.Resize(self.crop_size, interpolation=2)(warped_cloth)
        warped_cloth = self.transform(warped_cloth)
        warped_cloth_mask_name = im_name.replace("image", "cloth-warp-mask" if not self.unpaired else "unpaired-cloth-warp-mask")
        warped_cloth_mask = Image.open(osp.join(self.data_path, warped_cloth_mask_name))
        warped_cloth_mask = transforms.Resize(self.crop_size, interpolation=transforms.InterpolationMode.NEAREST)(warped_cloth_mask)
        warped_cloth_mask = self.toTensor(warped_cloth_mask)
        warped_cloth = warped_cloth * warped_cloth_mask

        # Remove the warp image
        feat = warped_cloth * (1 - inpaint_mask) + im * inpaint_mask
        feat_with_pose = im * inpaint_mask
        
        # down, up, left, right = mask2bbox(cm[key][0].numpy())
        # ref_image = c[key][:, down:up, left:right]
        ref_image = c[key]
        # ref_image = (ref_image + 1.0) / 2.0
        # ref_image = transforms.Resize((224, 224))(ref_image)
        # ref_image = self.clip_normalize(ref_image)

        # load pose points
        pose_name = im_name.replace("image", "openpose_json").replace(".jpg", "_keypoints.json")
        with open(osp.join(self.data_path, pose_name), "r") as f:
            pose_label = json.load(f)
            pose_data = pose_label["people"][0]["pose_keypoints_2d"]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(self.crop_size, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        tensor_to_image(agnostic, "./internal/agnostic.jpg")

        # load pose image
        pose_im_name = im_name.replace("image", "openpose_img").replace(".jpg", "_rendered.png")
        pose_im = Image.open(osp.join(self.data_path, pose_im_name))
        # pose_im = pose_im.convert("RGB")
        pose_im = transforms.Resize(self.crop_size, interpolation=0)(pose_im)
        # Convert black background to transparency
        datas = pose_im.getdata()
        newData = []
        for item in datas:
            # Change all pixels that are completely black to be transparent
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((128, 128, 128))
            else:
                newData.append(item)

        pose_im.putdata(newData)
        pose_im = self.transform(pose_im)
        tensor_to_image(pose_im, "./internal/pose_im.jpg")

        # add openpose into inpaint image
        feat_with_pose = feat_with_pose + pose_im * (1 - inpaint_mask)
        tensor_to_image(feat_with_pose, "./internal/feat_with_pose.jpg")

        # load image-parse-agnostic
        parse_name = im_name.replace("image", "image-parse-agnostic-v3.2").replace(".jpg", ".png")
        image_parse_agnostic = Image.open(osp.join(self.data_path, parse_name))
        image_parse_agnostic = transforms.Resize(self.crop_size, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        hands_mask = torch.sum(new_parse_agnostic_map[5:7], dim=0, keepdim=True)
        hands_mask = torch.clamp(hands_mask, min=0.0, max=1.0)

        inpaint_with_pose = feat_with_pose * (1 - hands_mask) + agnostic * hands_mask
        inpaint_warp_cloth = feat * (1 - hands_mask) + agnostic * hands_mask
        tensor_to_image(inpaint_with_pose, "./internal/inpaint_with_pose.jpg")
        tensor_to_image(inpaint_warp_cloth, "./internal/inpaint_warp_cloth.jpg")

        # get the caption of the cloth
        caption = self.caption_dict[c_name[key]]

        # get masked vton image by ootd
        c_img = np.array(c_img).astype(np.uint8)
        result = {
            "GT": im,
            # "inpaint_image": inpaint_warp_cloth,
            # "inpaint_image": im * inpaint_mask,
            "inpaint_image": agnostic,
            "inpaint_pa": ref_image_pa,
            "inpaint_mask": inpaint_mask,
            "mask": mask,
            "ref_imgs": ref_image,
            "warp_feat": feat,
            "file_name": self.im_names[index],
            "cloth_array": np.array(c_img).astype(np.uint8),
            "input_ids": "",
            # "prompt": caption,
            "prompt": "upperbody",
        }
        return result

    def __len__(self):
        return len(self.im_names)


class CPDatasetV2(CPDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.fine_height = 384
        # self.fine_width = 512
        # self.crop_size = (self.fine_height, self.fine_width)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def name(self):
        return "CPDatasetV2"
    
    def __getitem__(self, index):
        im_name = self.im_names[index]
        im_name = "image/" + im_name
        c_name = {}
        c = {}
        cm = {}
        if self.unpaired:
            key = "unpaired"
        else:
            key = "paired"

        c_name[key] = self.c_names[key][index]
        c[key] = Image.open(osp.join(self.data_path, "cloth", c_name[key])).convert("RGB")
        c[key] = transforms.Resize(self.crop_size, interpolation=2)(c[key])
        c_img = c[key]
        cm[key] = Image.open(osp.join(self.data_path, "cloth-mask", c_name[key]))
        cm[key] = transforms.Resize(self.crop_size, interpolation=0)(cm[key])
        cm_img = cm[key]

        cm_array = np.array(cm[key])
        cm_array = (cm_array >= 128).astype(np.float32)
        cm[key] = torch.from_numpy(cm_array)  # [0,1]
        cm[key].unsqueeze_(0)

        # person image
        im_pil_big = Image.open(osp.join(self.data_path, im_name))
        im_pil = transforms.Resize(self.crop_size, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)

        # load parsing image
        parse_name = im_name.replace("image", "image-parse-v3").replace(".jpg", ".png")
        im_parse_pil_big = Image.open(osp.join(self.data_path, parse_name))
        im_parse_pil = transforms.Resize(self.crop_size, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()

        # parse map
        labels = {
            0: ["background", [0, 10]],
            1: ["hair", [1, 2]],
            2: ["face", [4, 13]],
            3: ["upper", [5, 6, 7]],
            4: ["bottom", [9, 12]],
            5: ["left_arm", [14]],
            6: ["right_arm", [15]],
            7: ["left_leg", [16]],
            8: ["right_leg", [17]],
            9: ["left_shoe", [18]],
            10: ["right_shoe", [19]],
            11: ["socks", [8]],
            12: ["noise", [3, 11]],
        }

        parse_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_map[i] += parse_map[label]

        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i

        mask_id = torch.Tensor([3, 5, 6])
        mask = torch.isin(parse_onehot[0], mask_id).numpy()

        kernel_size = int(5 * (self.fine_width / 256))
        mask = cv2.dilate(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=3)
        mask = cv2.erode(mask.astype(np.uint8), kernel=np.ones((kernel_size, kernel_size)), iterations=1)
        mask = mask.astype(np.float32)
        inpaint_mask = 1 - self.toTensor(mask)

        ref_image = c[key]
        ref_image = self.transform(ref_image)

        # load pose points
        pose_name = im_name.replace("image", "openpose_json").replace(".jpg", "_keypoints.json")
        with open(osp.join(self.data_path, pose_name), "r") as f:
            pose_label = json.load(f)
            pose_data = pose_label["people"][0]["pose_keypoints_2d"]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))[:, :2]
        agnostic = self.get_agnostic(im_pil_big, im_parse_pil_big, pose_data)
        agnostic = transforms.Resize(self.crop_size, interpolation=2)(agnostic)
        agnostic = self.transform(agnostic)

        # load pose image
        pose_im_name = im_name.replace("image", "openpose_img").replace(".jpg", "_rendered.png")
        pose_im = Image.open(osp.join(self.data_path, pose_im_name))
        pose_im = transforms.Resize(self.crop_size, interpolation=0)(pose_im)
        # Convert black background to transparency
        datas = pose_im.getdata()
        newData = []
        for item in datas:
            # Change all pixels that are completely black to be transparent
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((128, 128, 128))
            else:
                newData.append(item)

        # load image-parse-agnostic
        parse_name = im_name.replace("image", "image-parse-agnostic-v3.2").replace(".jpg", ".png")
        image_parse_agnostic = Image.open(osp.join(self.data_path, parse_name))
        image_parse_agnostic = transforms.Resize(self.crop_size, interpolation=0)(image_parse_agnostic)
        parse_agnostic = torch.from_numpy(np.array(image_parse_agnostic)[None]).long()
        parse_agnostic_map = torch.FloatTensor(20, self.fine_height, self.fine_width).zero_()
        parse_agnostic_map = parse_agnostic_map.scatter_(0, parse_agnostic, 1.0)
        new_parse_agnostic_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                new_parse_agnostic_map[i] += parse_agnostic_map[label]
        hands_mask = torch.sum(new_parse_agnostic_map[5:7], dim=0, keepdim=True)
        hands_mask = torch.clamp(hands_mask, min=0.0, max=1.0)
        
        # load captions
        caption_name = osp.join(self.data_path, "cloth", c_name[key]).replace("cloth", "cloth_caption").replace(".jpg", ".txt")
        # Check if the file exists
        if os.path.exists(caption_name):
            with open(caption_name, 'r') as file:
                caption_string = file.read()
        else:
            print("File does not exist. ", caption_name)
            caption_string = "A cloth"  # Set caption_string to an empty string or handle the case when the file doesn't exist


        # get masked vton image by ootd
        c_img = np.array(c_img).astype(np.uint8)
        result = {
            "GT": im,
            # "inpaint_image": inpaint_warp_cloth,
            # "inpaint_image": im * inpaint_mask,
            "inpaint_image": agnostic,
            "inpaint_mask": inpaint_mask,
            "mask": mask,
            "ref_imgs": ref_image,
            "file_name": self.im_names[index],
            "prompt": caption_string,
            # "prompt": "upperbody",
        }
        return result


def pre_alignment(c, cm, parse_roi):
    align_factor = 1.0
    w, h = c.size

    # flat-cloth forground & bbox
    c_array = np.array(c)
    cm_array = np.array(cm)
    c_fg = np.where(cm_array != 0)
    t_c, b_c = min(c_fg[0]), max(c_fg[0])
    l_c, r_c = min(c_fg[1]), max(c_fg[1])
    c_bbox_center = [(l_c + r_c) / 2, (t_c + b_c) / 2]
    c_bbox_h = b_c - t_c
    c_bbox_w = r_c - l_c

    # parse-cloth forground & bbox
    parse_roi_fg = np.where(parse_roi != 0)
    t_parse_roi, b_parse_roi = min(parse_roi_fg[0]), max(parse_roi_fg[0])
    l_parse_roi, r_parse_roi = min(parse_roi_fg[1]), max(parse_roi_fg[1])
    parse_roi_center = [(l_parse_roi + r_parse_roi) / 2, (t_parse_roi + b_parse_roi) / 2]
    parse_roi_bbox_h = b_parse_roi - t_parse_roi
    parse_roi_bbox_w = r_parse_roi - l_parse_roi

    # scale_factor & paste location
    if c_bbox_w / c_bbox_h > parse_roi_bbox_w / parse_roi_bbox_h:
        ratio = parse_roi_bbox_h / c_bbox_h
        scale_factor = ratio * align_factor
    else:
        ratio = parse_roi_bbox_w / c_bbox_w
        scale_factor = ratio * align_factor
    paste_x = int(parse_roi_center[0] - c_bbox_center[0] * scale_factor)
    paste_y = int(parse_roi_center[1] - c_bbox_center[1] * scale_factor)

    # cloth alignment
    c = c.resize((int(c.size[0] * scale_factor), int(c.size[1] * scale_factor)), Image.BILINEAR)
    blank_c = Image.fromarray(np.ones((h, w, 3), np.uint8) * 255)
    blank_c.paste(c, (paste_x, paste_y))
    c = blank_c  # PIL Image
    # c.save(os.path.join(cloth_align_dst, cname))

    # cloth mask alignment
    cm = cm.resize((int(cm.size[0] * scale_factor), int(cm.size[1] * scale_factor)), Image.NEAREST)
    blank_cm = Image.fromarray(np.zeros((h, w), np.uint8))
    blank_cm.paste(cm, (paste_x, paste_y))
    cm = blank_cm  # PIL Image
    # cm.save(os.path.join(clothmask_align_dst, cmname))
    return c, cm


if __name__ == "__main__":
    dataset = CPDataset("/data/user/gjh/VITON-HD", 512, mode="train", unpaired=False)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    for data in loader:
        pass