from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar_data():
    normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train,
    )
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test,
    )

    return train_dataset, test_dataset

def get_dataloaders(batch_size, num_workers):
    train_dataset, test_dataset = get_cifar_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
