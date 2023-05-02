import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[np.ndarray, np.array]:
    """
    Compute the mean and the standard deviation of the pixel values in the dataset.

    Note: convert the image in grayscale and then scale to [0,1] before computing
    mean and standard deviation

    Hints: use StandardScalar (check import statement)

    Args:
    -   dir_name: the path of the root dir
    Returns:
    -   mean: mean value of the dataset (np.array containing a scalar value)
    -   std: standard deviation of th dataset (np.array containing a scalar value)
    """

    mean = None
    std = None

    ############################################################################
    # Student code begin
    ############################################################################

    shape = (-1, 1)

    scaler = StandardScaler()
    for directory1 in os.listdir(dir_name):
        path1 = os.path.join(dir_name, directory1)
        for directory2 in os.listdir(path1):
            path2 = os.path.join(path1, directory2)
            for directory3 in os.listdir(path2):
                data_path = os.path.join(path2, directory3)
                img = Image.open(data_path)
                #grayscalse_img = img.convert('GSC')
                #transforms.Grayscale(num_output_channels = 1),
                grayscalse_img = np.array(img.convert('L')).astype('float32')
                #print(grayscalse_img)
                scaled_img = grayscalse_img / 255
                #print(scaled_img)
                stat_data = np.reshape(scaled_img.flatten(), shape)
                scaler.partial_fit(stat_data)
                
    mean = scaler.mean_
    std = scaler.scale_

    ############################################################################
    # Student code end
    ############################################################################
    return mean, std
