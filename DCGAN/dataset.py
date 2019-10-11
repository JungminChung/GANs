import torch, os
from PIL import Image
from torch.utils.data.dataset import Dataset

class celebDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.name_list = os.listdir(self.img_path)

    def __getitem__(self, index):
        image_name = self.name_list[index]

        img = Image.open(os.path.join(self.img_path, image_name))

        if self.transform : 
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.name_list)