import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar_data():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

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

    return transform_train, transform_test, train_dataset, test_dataset


def get_dataloaders(batch_size, num_workers):
    transform_train, transform_test, train_dataset, test_dataset = get_cifar_data()

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


# -----------------------
# Training & Eval Functions
# -----------------------
def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # if (batch_idx + 1) % 100 == 0:
        #     print(
        #         f"Epoch [{epoch}] "
        #         f"Step [{batch_idx+1}/{len(train_loader)}] "
        #         f"Loss: {loss.item():.4f}"
        #     )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    # print(f"Train Epoch {epoch}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}")

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    # print(f"Test : Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}")
    return epoch_loss, epoch_acc
