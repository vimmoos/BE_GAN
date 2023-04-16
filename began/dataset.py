import torch.utils.data as data
import os

import cv2
import torchvision.transforms as transforms


class CelebA(data.Dataset):
    def __init__(
        self,
        data_path="data/32_32_crop/",
        size=64,
        load_all=False,
    ):
        super().__init__()
        data_path = os.path.abspath(data_path)
        self.image_list = [
            os.path.join(data_path, x)
            for x in os.listdir(data_path)
            if any(
                x.endswith(extension)
                for extension in [".png", ".jpg", ".jpeg"]
            )
        ]
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((size, size)),
            ]
        )
        self.images = None
        if load_all:
            self.images = [
                self.transforms(cv2.imread(path)) for path in self.image_list
            ]

    def __getitem__(self, index):
        if self.images:
            return self.images[index]
        path = self.image_list[index]
        return self.transforms(cv2.imread(path))

    def __len__(self):
        return len(self.image_list)
