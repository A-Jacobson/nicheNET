import os
import glob

import torch
import torch.utils.data as data

from nichenet import config


class RasterData(data.Dataset):
    def __init__(self, root, train=True):
        self.root = root
        self.train = train
        pattern = os.path.join(root, "test", "*.pt")

        if self.train:
            pattern = os.path.join(root, "train", "*.pt")

        self.data_files = glob.glob(pattern)

    def __getitem__(self, index):
        file_path = self.data_files[index]
        raster, name = torch.load(file_path)
        raster = raster.float()
        raster -= config.MEAN
        raster /= config.STD
        target = int(name[-1])
        return raster, target

    def __len__(self):
        return len(self.data_files)

