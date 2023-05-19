# This code is largely inspired by torchvision.datasets.MNIST

from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings
from PIL import Image
import numpy as np
import os
import pathlib

from tqdm import tqdm
import torch
from torchvision.datasets import VisionDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms


#transforms for the tiny-imagenet dataset. Applicable for the tasks 1-4
data_transforms_tin = {
	'train': transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(28),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'test': transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(28),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}


#transforms for the mnist dataset. Applicable for the tasks 5-9
data_transforms_mnist = {
	'train': transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(28),
		transforms.ToTensor(),
		transforms.Normalize([0.1307,], [0.3081,])
	]),
	'test': transforms.Compose([
		transforms.Resize(32),
		transforms.CenterCrop(28),
		transforms.ToTensor(),
		transforms.Normalize([0.1307,], [0.3081,])
	])
}

class TINMNIST(VisionDataset):
    data = []
    targets = []
    classes = []
    
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {_class: i for i, _class in enumerate(self.classes)}

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(TINMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set
        mode = 'train' if train else 'test'

        #if self._check_legacy_exist():
        #    self.data, self.targets = self._load_legacy_data()
        #    return

        #if not self._check_exists():
        #    raise RuntimeError('Dataset not found.' +
        #                       ' You can use download=True to download it')
        
        workDir  = pathlib.Path().resolve()
        dataDir  = os.path.join(workDir, 'TINMNIST')

        #download dataset maybe sometime

        #load self.data and self.targets
        self.data = np.zeros((180000, 3, 28, 28))
        self.targets = np.zeros(180000)
        i = 0
        label_offset = -50
        for task_num in range(1, 9+1):
            print("\nLoading task " + str(task_num) + " of 9")
            path_task = os.path.join(dataDir, "Task_" + str(task_num))
            
            image_folder = None
            if (task_num <= 4):
                image_folder = datasets.ImageFolder(os.path.join(path_task, mode), transform = data_transforms_tin[mode])
                label_offset += 50
            else:
                image_folder = datasets.ImageFolder(os.path.join(path_task, mode), transform = data_transforms_mnist[mode])
                if (task_num == 5):
                    label_offset += 50
                else:
                    label_offset += 2
            
            dset_size = len(image_folder)
            dset_loaders = torch.utils.data.DataLoader(image_folder, batch_size = 1024, shuffle=False, num_workers=1)

            #load data within that folder
            for data, labels in tqdm(dset_loaders):
                pre_i = i
                i += len(data)
                self.data[pre_i:i, :, :, :] = data
                self.targets[pre_i:i] = torch.add(labels, label_offset)

        print("\nResizing data structure")
        self.data = np.resize(self.data, (i, 3, 28, 28))
        self.targets = np.resize(self.targets, i)
        #print("TINMNIST loaded")
        self.classes = [i for i in range(int(max(self.targets)+1))]

        
    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        img = np.transpose(img, [1, 2, 0])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target