import os
from glob import glob
from torch.utils.data import Dataset
from utils import stack_layers
import torch


class ASCFolder(Dataset):
    def __init__(self, path, num_layers, transform=None, limit=None):
        self.data_files = glob(os.path.join(path, '*', '*'))
        if limit:
            self.data_files = self.data_files[:limit//2].append(self.data_files[:-limit//2])
        self.labels = [os.path.split(label)[-1] for label in glob(os.path.join(path, '*'))]
        self.labels2idx = {label: idx for idx, label in enumerate(sorted(self.labels))}
        self.num_layers = num_layers
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.data_files[index]
        target = self.labels2idx[file_path.split('/')[-2]]
        raster = torch.from_numpy(stack_layers(file_path, self.num_layers)).float()
        if self.transform:
            raster = self.transform(raster)
        return raster, torch.FloatTensor([target])

    def __len__(self):
        return len(self.data_files)
