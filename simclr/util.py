import torch
from torchvision import transforms as T


def get_simclr_transform():
    return T.Compose(
        [
            T.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
        ]
    )


def get_eval_transform():
    return T.Compose(
        [
            T.ToTensor(),
        ]
    )


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, target = self.base_dataset[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return (x1, x2, target)
