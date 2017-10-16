import os
from torch.autograd import Variable
import numpy as np


def load_asc(asc_file):
    return np.loadtxt(asc_file, skiprows=6)


def stack_layers(raster_folder, num_layers):
    return np.array([load_asc(os.path.join(raster_folder, raster_folder.split('/')[-1] + "_" + str(i) + ".asc"))
                     for i in range(1, num_layers + 1)])


def min_max_scale(img, data_min, data_max):
    """
    data_min = [0.0, 0.0, -10.3, -8.3, -13.8, 0.0, 0.0]
    data_max = [290.0, 18692.0, 15.9, 24.6, 12.1, 1.16, 7.1]
    """
    for c in range(len(img)):
        img[c, :, :] = ((img[c, :, :] - data_min[c]) /
                        (data_max[c] - data_min[[c]]))
    return img


def change_nodata_val(x, new_val=0, current_val=-3.4e+38):
    x[x == current_val] = new_val
    return x


def prep_sample(x):
    return Variable(x.unsqueeze(0).cuda())
