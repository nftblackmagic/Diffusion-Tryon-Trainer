conda create -n ootd python==3.10
conda activate ootd
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 numpy==1.24.4 scipy==1.10.1 scikit-image==0.21.0 opencv-python==4.7.0.72 pillow==9.4.0 diffusers==0.24.0 transformers==4.36.2 accelerate==0.26.1 matplotlib==3.7.4 tqdm==4.64.1 gradio==4.16.0 config==0.5.1 einops==0.7.0 ninja==1.10.2
pip install diffusers==0.26.3
pip install --upgrade huggingface_hub
pip install basicsr datasets xformers bitsandbytes wandb