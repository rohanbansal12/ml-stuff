"""
Data loading utilities for CIFAR-10.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar_data(data_dir: str = "./data"):
    """
    Load CIFAR-10 with normalization to [-1, 1].

    DDPM typically works in [-1, 1] range:
    - Forward process adds Gaussian noise
    - Reverse process predicts to recover clean image in [-1, 1]
    """
    # Normalize to [-1, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Simple augmentation
            transforms.ToTensor(),
            normalize,
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train,
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test,
    )

    return train_dataset, test_dataset


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    persistent_workers: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test data loaders."""
    train_dataset, test_dataset = get_cifar_data(data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Avoid small final batches
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=3 if num_workers > 0 else None,
    )

    return train_loader, test_loader
