from typing import Tuple, List, Callable
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SimpleDataset(Dataset):
    """A Dataset class that pairs features and labels.
    Provides the method apply() for custom preprocessing functions.

    Args:
        features (list): list of input features.
        labels (list): list of target labels.
    """

    def __init__(self, features: List[np.ndarray], labels: List[np.ndarray]):
        super().__init__()
        assert len(features) == len(labels), "feature size must match label size"
        self.features = features
        self.labels = labels

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)

    def apply(self, func: Callable[[np.ndarray], np.ndarray]):
        """Apply a preprocessing function to features.

        Args:
            func (Callable): function that takes array and returns mapped array.

        Returns:
            self
        """
        self.features = [func(x) for x in self.features]
        return self


def build_train_test_dataloaders(
        x: List[np.ndarray],
        y: List[np.ndarray],
        batch_size: int = 1,
        split_ratio: float = 0.2,
        stratify: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
    """Generates train and test dataloader from given feature-label pairs.

    Args:
        x (list): list of input features. Expected shape: (N, C, H, W)
        y (list): list of target labels. Expected shape: (N, )
        batch_size (int, optional): batch size. Defaults to 1.
        split_ratio (float, optional): the portion of test size. Defaults to 0.2.
        stratify (bool, optional): equalize label distribution of train and test set. Defaults to True.

    Returns:
        Tuple[DataLoader, DataLoader]: both dataloaders for training set and test set.
    """
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=split_ratio,
                                                        stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=split_ratio,
                                                        stratify=None)

    # define tranformations for preprocessing feature vectors
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))
    ])

    # define dataset
    train_dataset = SimpleDataset(X_train, y_train).apply(transform_train)
    test_dataset = SimpleDataset(X_test, y_test).apply(transform_test)

    # define dataloader
    train_dataloader = DataLoader(train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=0,
                                collate_fn=None,
                                pin_memory=False,
                                drop_last=False
                                )
    test_dataloader = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=None,
                                pin_memory=False,
                                drop_last=False
                                )

    return train_dataloader, test_dataloader

def run_example():
    print("=== Generate example dataset ===")
    N = 1000 # size of dataset
    N_CLASS = 10 # number of class categories
    BATCH_SIZE = 64
    TEST_SPLIT_RATIO = 0.25
    X_SHAPE = (N, 28, 28, 3)
    Y_SHAPE = (N, )

    x = np.random.randint(0, 256, X_SHAPE, np.uint8)
    y = np.random.randint(0, N_CLASS, Y_SHAPE, np.int32)
    print(f"Dataset shape: {tuple(x.shape)}, {tuple(y.shape)}")

    print("====== Create dataloaders ======")
    train_dataloader, test_dataloader = build_train_test_dataloaders(x, y,
                                            batch_size=BATCH_SIZE,
                                            split_ratio=TEST_SPLIT_RATIO)
    print(f"Batch count: Trainset {len(train_dataloader)}, Testset {len(test_dataloader)}")

    print("========= Draw batches =========")
    for x, y in train_dataloader:
        print(f"Trainset batch: {tuple(x.shape)}, {tuple(y.shape)}")
    for x, y in test_dataloader:
        print(f"Testset batch: {tuple(x.shape)}, {tuple(y.shape)}")

if __name__ == "__main__":
    run_example()