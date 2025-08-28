import os, numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegDataset(Dataset):
    def __init__(self, img_dir, msk_dir, size=(512,512), aug=True):
        self.imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
        self.img_dir, self.msk_dir = img_dir, msk_dir
        t = [A.Resize(*size)]
        if aug: t += [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2)]
        t += [ToTensorV2()]
        self.tf = A.Compose(t)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, i):
        name = os.path.splitext(self.imgs[i])[0]
        img = np.array(Image.open(os.path.join(self.img_dir, name + '.png')).convert('RGB'))
        msk = np.array(Image.open(os.path.join(self.msk_dir,  name + '.png')).convert('L'))  # maska jako label id [0..K]
        out = self.tf(image=img, mask=msk)
        x = out['image'].float()/255.0
        y = out['mask'].long()
        return x, y