import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms 
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pandas as pd
import glob
import cv2
from PIL import Image
import numpy as np


class DomeMaster(Dataset):
    def __init__(self, dir, train=True, train_split=0.80):
        assert os.path.exists(dir)
        self.files = glob.glob(os.path.join(dir, "*png"))
        self.files = sorted(self.files)
        self.transform = transforms.Compose([
    transforms.Resize((368,368)),
    transforms.ToTensor(),
])
  
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rgb = self.files[idx]
        image = Image.open(rgb)
        image = self.transform(image)
        return image, '', '' 



if __name__ == "__main__":
    ds = XRegoDataset(dir="/home/javaid/Desktop/Thesis/Datasets/xrEgoPose_trainset_male_010_f_s_env_001_000")
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    for image, depth, label in dl:

        print (label.shape, depth.shape)
