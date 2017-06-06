import glob
import os

import numpy as np
import torch

from nichenet import config


def load_asc(asc_file):
    return np.loadtxt(asc_file, skiprows=6)


def stack_layers(raster_folder, num_layers):
    return np.array([load_asc(os.path.join(raster_folder, raster_folder.split('/')[-1] + "_" + str(i) + ".asc"))
                     for i in range(1, num_layers + 1)])


def raster_image_generator(data_path, num_layers=config.NUM_LAYERS):
    raster_folders = glob.glob(os.path.join(config.DATA_PATH, data_path, '*', '*'))
    for raster_folder in raster_folders:
        path_list = raster_folder.split("/")
        if path_list[-2] == 'true':
            label = '1'
        else:
            label = '0'
        raster_image = stack_layers(raster_folder, num_layers)
        label = "_".join((path_list[-1], label))
        if raster_image.shape != (8, 11, 11):
            print('bad raster', label)
            continue
        yield raster_image, label


def save_rasters_as_tensors(data_path, out_path='raster_tensors'):
    for raster, label in raster_image_generator(data_path):
        raster_tensor = torch.from_numpy(raster), label
        torch.save(raster_tensor, os.path.join(config.DATA_PATH, out_path, label+'.pt'))
