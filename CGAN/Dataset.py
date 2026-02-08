# Dataset.py

import os
import numpy as np
import pandas as pd
from skimage import io
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F


class CustomDataset(Dataset):
    def __init__(self, folderPath, is_train=True, output_size=512):

        csv_path = os.path.join(folderPath, 'chairs.train.class.csv' if is_train else 'chairs.valid.class.csv')
        print(f"Loading dataset from {csv_path}")

        self.data_info = pd.read_csv(csv_path)

        old_prefix = ""
        new_prefix = ""
        self.images = self.data_info['image_path'].values
        self.images = np.array([
            str(path).replace(old_prefix, new_prefix) for path in self.images
        ])

        self.blowing_ratios = self.data_info['blowing_ratio'].values
        self.mist_concentrations = self.data_info['mist_concentration'].values
        self.drop_diameters = self.data_info['drop_diameter'].values

        print('Loaded ', len(self.images), ' files')

        self.is_train = is_train
        self.output_size = output_size

        self.transformations = transforms.Compose([
            transforms.Resize((output_size, output_size)),
            transforms.ToTensor()
        ])

        self.norm_params = {
            'br_max': 2.0,
            'mc_max': 10.0,
            'dd_max': 50.0
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_filename = self.images[index]

        try:

            image = io.imread(image_filename)
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            image = F.to_pil_image(image)


            image = self.transformations(image)

            br_val = float(self.blowing_ratios[index])
            mc_val = float(self.mist_concentrations[index])
            dd_val = float(self.drop_diameters[index])

            br_norm = br_val / (self.norm_params['br_max'] + 1e-8)
            mc_norm = mc_val / (self.norm_params['mc_max'] + 1e-8)
            dd_norm = dd_val / (self.norm_params['dd_max'] + 1e-8)

            blowing_ratio = torch.tensor([br_norm], dtype=torch.float32)
            mist_concentration = torch.tensor([mc_norm], dtype=torch.float32)
            drop_diameter = torch.tensor([dd_norm], dtype=torch.float32)

            return image, blowing_ratio, mist_concentration, drop_diameter

        except Exception as e:
            print(f"Error loading sample {index}: {e}  |  image: {image_filename}")

            return (torch.zeros(3, self.output_size, self.output_size),
                    torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))