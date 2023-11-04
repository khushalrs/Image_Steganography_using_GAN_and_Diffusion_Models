import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MapDataset(Dataset):
    def __init__(self, root_dir_cover, root_dir_hidden):
        self.root_dir_cover = root_dir_cover
        self.list_files_cover = os.listdir(self.root_dir_cover)
        self.root_dir_hidden = root_dir_hidden
        self.list_files_hidden = os.listdir(self.root_dir_hidden)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file_cover = self.list_files_cover[index]
        img_path_cover = os.path.join(self.root_dir_cover, img_file_cover)
        cover = np.array(Image.open(img_path_cover))
        img_file_hidden = self.list_files_hidden[index]
        img_path_hidden = os.path.join(self.root_dir_hidden, img_file_hidden)
        hidden = np.array(Image.open(img_path_hidden))

        transform = A.Compose(
            [A.Resize(width=256, height=256),
             A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,), ToTensorV2(),],
        )

        cover = transform(input=cover)["image"]
        hidden = transform(input=hidden)["image"]

        return cover, hidden
