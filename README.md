# Oxford Flowers – Deep Learning Models (CNN, VAE, UNet)
This repository showcases deep learning models built using PyTorch for the Oxford Flowers dataset. It includes:

- CNN classifiers (coarse and fine grained)
- Variational Autoencoder (VAE) for image reconstruction
- UNet for image latent representation denoising

# Trained Model Performance
 Coarse CNN 100 epochs| Classification (coarse labels)   | **85.13%** test accuracy             | ~452K
 
 Fine CNN 100 epochs  | Classification (fine labels)     | **83.63%** test accuracy             | ~499K 
 
 VAE 60 epochs        | Image Reconstruction             | Per-pixel error: 0.006244 ± 0.002823 | ~6.3M 
 
 UNet 110 epochs       | Denoising Latent Representation  | Train Loss: 0.9388, Val Loss: 0.9418 | ~27.77M 

Note: `UNet_110.pth` is not included due to GitHub’s file size limits (>100MB). You can retrain from scratch using the provided scripts.

# Installation Requirements
pip install torch torchvision numpy tqdm seaborn scikit-learn

*For more details of model architecture and training/validation/testing, a technical report is uploaded. 
