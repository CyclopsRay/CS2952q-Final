import torch
from torchvision import transforms
from torch.utils.data import Dataset
import os
import PIL


class PizzaDataset(Dataset):

    def __init__(self, path):
        super(PizzaDataset, self).__init__()
        self.path = path
        self.image_paths = []
        paths = os.listdir(path)

        for p in paths:
            self.image_paths.append(os.path.join(path, p))


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = transforms.ToTensor()(PIL.Image.open(self.image_paths[idx]))
        return img