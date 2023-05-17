from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings
from PIL import Image
import numpy as np


from torchvision.datasets import VisionDataset


class TINMNIST(VisionDataset):
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

        #if self._check_legacy_exist():
        #    self.data, self.targets = self._load_legacy_data()
        #    return

        #if download:
        #    self.download()

        #if not self._check_exists():
        #    raise RuntimeError('Dataset not found.' +
        #                       ' You can use download=True to download it')

        #self.data, self.targets = self._load_data()

        
    def __len__(self) -> int:
        return len(self.data)

    data = [[]]
    targets = [[]]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target