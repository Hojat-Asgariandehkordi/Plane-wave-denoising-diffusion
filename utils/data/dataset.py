import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat


class ANGLES(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data_dir = root_dir
        self.lbl1 = '1'
        self.lbl2 = '75'
        self.image_list = os.listdir(os.path.join(root_dir, self.lbl1))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_dir = os.path.join(self.data_dir, self.lbl1, self.image_list[idx])
        image = loadmat(img_dir)['DASRFs'][:1072, :]
        label = 37
        image2 = loadmat(os.path.join(self.data_dir, self.lbl2, self.image_list[idx]))['DASRFs'][:1072, :]
        image1, das_bmode1 = self._process_image(image)
        image2_2, das_bmode2 = self._process_image(image2)
        input1 = np.concatenate((image1, das_bmode1, das_bmode2), axis=2)
        input2 = np.concatenate((image2_2, das_bmode2, das_bmode2), axis=2)

        if self.transform:
            input1, input2 = self.transform(input1), self.transform(input2)
        label = torch.tensor(37 - label)

        return input1, input2, label

    def _process_image(self, image):
        image = image.reshape((1072, 192, 1))
        das_env = np.abs(np.fft.ifft(np.fft.fft(image) * 1j))
        das_bmode = 20 * np.log10(das_env / np.max(das_env) + 0.001)
        return image, das_bmode
