# Multi-class Image Segmentation with U-Net (Demo)

This repository contains **a simplified fragment** of my work on deep learning models for image segmentation.  
The full project is part of my volunteer research at the **Foundation for Cardiac Surgery Development**, where I contribute to an AI-based vision system for a robotic surgical platform.  
Due to confidentiality, I cannot share the complete solution, but I am able to provide this demo to illustrate my coding style and technical skills.  

---

## What's included
- **U-Net implementation from scratch** in PyTorch (`model.py`)  
- **Custom dataset class** with Albumentations augmentations (`dataset.py`)  
- **Combined loss function** (Cross Entropy + Dice) for multi-class segmentation (`losses.py`)  
- **Training pipeline** with AMP (automatic mixed precision), learning rate scheduling, and best-checkpoint saving (`train.py`)  

---

## Data
The demo assumes a dataset structure with separate folders for images and segmentation masks: 
data/
train/
images/
masks/
val/
images/
masks/

---

## Example
<img width="494" height="284" alt="Screenshot 2025-05-18 at 7 27 33 PM" src="https://github.com/user-attachments/assets/39119603-052d-4701-b9d6-684cdfc945ca" /><img width="490" height="279" alt="Screenshot 2025-05-18 at 7 27 39 PM" src="https://github.com/user-attachments/assets/a763fad9-c665-4a9b-b7ad-b21bc1d2bd15" />

