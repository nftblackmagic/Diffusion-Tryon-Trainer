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
