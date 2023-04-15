import torch.utils.data as data
import os

import cv2
import torchvision.transforms as transforms


def is_image_file(filename):
    return any(
        filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"]
    )


class CelebA(data.Dataset):
    def __init__(
        self,
        data_path="data/64_64_crop/",
        size=64,
    ):
        super().__init__()
        data_path = os.path.abspath(data_path)
        self.image_list = [
            os.path.join(data_path, x)
            for x in os.listdir(data_path)
            if is_image_file(x)
        ]
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((size, size)),
            ]
        )

    def __getitem__(self, index):
        path = self.image_list[index]
        return self.transforms(cv2.imread(path))

    def __len__(self):
        return len(self.image_list)
