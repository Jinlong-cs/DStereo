from typing import List, Optional, Union

import cv2
import msgpack
import numpy as np
import torch.utils.data as data

from PIL import Image

from hat.registry import OBJECT_REGISTRY

__all__ = ["VLADdescriptor"]


@OBJECT_REGISTRY.register
class VLADdescriptor(data.Dataset):
    """
    Dataset which gets training data and ground truth from a text file.

    Args:
        data_file: The path to the text file containing training data and ground truth paths.
        transforms: Optional transforms to be applied on a sample.
    """

    def __init__(self, data_file: str, transforms: list = None):
        self.data_file = data_file
        self.transforms = transforms
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        with open(self.data_file, 'r') as file:
            for line in file:
                training_path, truth_path = line.strip().split()
                samples.append((training_path, truth_path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = {}
        image_path, truth_path = self.samples[idx]
        
        # Load and convert the image to RGB
        image = Image.open(image_path).convert('RGB')

        # Load the ground truth
        gt = np.load(truth_path)
        
        data['img'] = image
        data['global_descriptor'] = gt['global_descriptor']
        
        if self.transforms:
            data = self.transforms(data)
        
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(data_file={self.data_file})'