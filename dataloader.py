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


class XRegoDataset(Dataset):
    def __init__(self, dir, train=True, train_split=0.80):
        assert os.path.exists(dir)
        files = glob.glob(os.path.join(dir, "*png"))
        files = sorted(files)
       
        file_ids = {i.split(".")[0].split("_")[-1]: {} for i in files}
        labels = glob.glob(os.path.join(dir, "*2dp.csv"))

        assert len(labels) == len(file_ids), (len(labels), len(file_ids))
        assert len(labels) > 1
        files.extend(labels) 

        for file in files:
            id = file.split(".")[0].split("_")[-1]
            assert os.path.exists(file)
            if "pose_image." in file:
                file_ids[id]["rgb"] = file
            elif "depth" in file:
                file_ids[id]["depth"] = file 
            elif ".csv" in file:
                file_ids[id]["label"] = file 
        self.xr_files = file_ids
        self.ids = list(self.xr_files.keys())
        total_samples = int(train_split * len(self.xr_files))
        if train:
            keys = self.ids[:total_samples]
            self.ids = self.ids[:total_samples]
        else:
            keys = self.ids[total_samples:]
            self.ids = self.ids[total_samples:]
        self.xr_files = {i: self.xr_files[i] for i in keys}
        assert len(self.xr_files) == len(self.ids)
        print("Total Samples in Set:", len(self.ids))
        # print(self.ids)
        self.load_joints()

        self.transform_dict = {
        'rgb': transforms.Compose(
        [transforms.Resize((224, 224)), 
         transforms.ToTensor(),
         # transforms.Normalize(mean=[0.485, 0.456, 0.406],
         #                      std=[0.229, 0.224, 0.225]),
         ]),
        'depth': transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor()
         ])}


    def load_joints(self):
        for file in self.xr_files:
            label = self.xr_files[file]["label"]
            label_df = pd.read_csv(label, header=None).to_numpy().reshape(-1, 2)
            
            right_arm = label_df[14: 17]
            left_arm = label_df[22: 25] 
            self.xr_files[file]["label"] = np.vstack((right_arm,  left_arm))


    def __len__(self):
        return len(self.xr_files)

    def __getitem__(self, idx):
        pose_id = self.ids[idx]
        rgb = self.xr_files[pose_id]["rgb"]
        depth = self.xr_files[pose_id]["depth"]
        label = self.xr_files[pose_id]["label"].copy()

        image = Image.open(rgb)
        depth_image = Image.open(depth)
        shape = np.array(image).shape
        for i in range(len(label)):
            label[i][0] /= shape[1]
            label[i][1] /= shape[0]
        label = label.reshape(-1)
        # for joint in label.reshape(-1,2): 
        #     print(joint[0] * shape[1], joint[1] * shape[0])
        #     cv2.circle(image, (int(joint[0] *shape[1]), int(joint[1] * shape[0])), radius=10, color=(255, 255, 0), thickness=-1)
        # plt.imshow(image)
        # plt.show()

        image = self.transform_dict["rgb"](image)
        depth_image = self.transform_dict["depth"](depth_image)
        # print(label)
        return image, depth_image, label 



if __name__ == "__main__":
    ds = XRegoDataset(dir="/home/javaid/Desktop/Thesis/Datasets/xrEgoPose_trainset_male_010_f_s_env_001_000")
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    for image, depth, label in dl:

        print (label.shape, depth.shape)
