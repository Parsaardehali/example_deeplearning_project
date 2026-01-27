# %%
import torch
import math
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import v2
import numpy as np
from glob import glob
import os
import h5py
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.file_list = glob(os.path.join(path, "*.npz"))
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        input_img = torch.tensor(data['input'], dtype=torch.float32).to(device)
        target_img = torch.tensor(data['target'], dtype=torch.float32).to(device)
        return input_img, target_img
    
