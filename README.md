## This is a unoffical training code for [OOTDiffusion](https://github.com/levihsu/OOTDiffusion)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-5-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

This repository contains the training code for the OOTDiffusion project. We extend our gratitude to the contributions of OOTDiffusion and have built upon this foundation by utilizing Huggingface's Diffusors library to implement training on the VTON dataset for virtual try-on. Our project aims to enhance the accuracy and realism of virtual try-ons through cutting-edge diffusion model technology, providing users with a more authentic try-on experience.

## Some results

![Sample 1](/out_00069_00.jpg_hd_0.png)
![Sample 2](/out_00654_00.jpg_hd_0.png)

## Install requirements

### Python env

```
conda env create -f environment.yaml
conda activate groot

```

### Data Preparation

#### VITON-HD

1. Download [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset
2. Download pre-warped cloth image/mask from [Google Driver](https://drive.google.com/drive/folders/15cBiA0AoSCLSkg3ueNFWSw4IU3TdfXbO?usp=sharing) or [Baidu Cloud](https://pan.baidu.com/s/1ss8e_Fp3ZHd6Cn2JjIy-YQ?pwd=x2k9) and put it under your VITON-HD dataset
3. Download cloth captions from train [Google Driver](https://drive.google.com/file/d/1WfQUh1O3uuoASCvRm-O2vMp4JBHVWj9Z/view?usp=drive_link) test [Google Driver](https://drive.google.com/file/d/1OM2zBFFRZUOigzrNG-WHHdeer9MuGaqz/view?usp=drive_link)

After these, the folder structure should look like this (the unpaired-cloth\* only included in test directory):

```
├── VITON-HD
|   ├── test_pairs.txt
|   ├── train_pairs.txt
│   ├── [train | test]
|   |   ├── image
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-warp
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── cloth-warp-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── unpaired-cloth-warp
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
│   │   ├── unpaired-cloth-warp-mask
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]
|   |   ├── cloth_caption
│   │   │   ├── [000006_00.jpg | 000008_00.jpg | ...]

```

### How to run training

```
bash train_ootd.sh
```

## Inference over CP dataset

### Weight preparation

Train the weights or download a pretrained weight from [Huggingface](https://huggingface.co/xiaozaa/Diffusion-Tryon-Trainer)
The weights need to be put under checkpoints dir.

### Run inference

```
sh inference_test_dataset.sh
```

## Acknowledgements

Our unet code is directly from [OOTDiffusion](https://github.com/levihsu/OOTDiffusion). We also thank [DCI-VTON-Virtual-Try-On](https://github.com/bcmi/DCI-VTON-Virtual-Try-On), our dataset module depends on it.

This code is only for study and research.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/nftblackmagic"><img src="https://avatars.githubusercontent.com/u/108916776?v=4?s=100" width="100px;" alt="nftblackmagic"/><br /><sub><b>nftblackmagic</b></sub></a><br /><a href="https://github.com/nftblackmagic/Diffusion-Tryon-Trainer/commits?author=nftblackmagic" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/MoonBlvd"><img src="https://avatars.githubusercontent.com/u/16040099?v=4?s=100" width="100px;" alt="Yu (Brian) Yao"/><br /><sub><b>Yu (Brian) Yao</b></sub></a><br /><a href="https://github.com/nftblackmagic/Diffusion-Tryon-Trainer/commits?author=MoonBlvd" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Stevada"><img src="https://avatars.githubusercontent.com/u/64606061?v=4?s=100" width="100px;" alt="Stevada"/><br /><sub><b>Stevada</b></sub></a><br /><a href="https://github.com/nftblackmagic/Diffusion-Tryon-Trainer/commits?author=Stevada" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dingkwang"><img src="https://avatars.githubusercontent.com/u/10276784?v=4?s=100" width="100px;" alt="Dingkang Wang"/><br /><sub><b>Dingkang Wang</b></sub></a><br /><a href="https://github.com/nftblackmagic/Diffusion-Tryon-Trainer/commits?author=dingkwang" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/lXXXw"><img src="https://avatars.githubusercontent.com/u/22513666?v=4?s=100" width="100px;" alt="xiaoweilu"/><br /><sub><b>xiaoweilu</b></sub></a><br /><a href="https://github.com/nftblackmagic/Diffusion-Tryon-Trainer/commits?author=lXXXw" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
