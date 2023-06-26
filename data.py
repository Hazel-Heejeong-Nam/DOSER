import torch
import os
from PIL import Image

class custom_MNIST(torch.utils.data.Dataset):
    training_file = "training.pt"
    test_file = "test.pt"

    def __init__(self, root = './', train = True, transform=None):
        super().__init__()
        self.root = root  
        self.train = train  # training set(True) or test set(False)
        self.data, self.targets = self._load_data()
        self.transform = transform

        if self.train: 
          self._del_zeros()

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.root, image_file))
        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.root, label_file))
        return data, targets
    
    def _del_zeros(self):
        idx = self.targets != 0
        self.data = self.data[idx]
        self.targets = self.targets[idx]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.numpy()
        if self.transform is not None:
          img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)
