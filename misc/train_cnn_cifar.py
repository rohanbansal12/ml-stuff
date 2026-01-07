import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------
# 1. Config
# -----------------------
batch_size = 128
num_epochs = 10
learning_rate = 1e-3
num_workers = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------
# 2. Data (CIFAR-10)
# -----------------------
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


# -----------------------
# 3. Model
# -----------------------
class MyBasicCNN(nn.Module):
    def __init__(self):
        super().__init__()

        ## dims 3x32x32 -> 128x8x8
        self.features = nn.Sequential(
            *[
                nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),  ## dims 3x32x32 -> 32x32x32
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    32, 64, kernel_size=3, padding=1, bias=False
                ),  ## dims 32x32x32 -> 64x32x32
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  ## dims 64x32x32 -> 64x16x16
                nn.Conv2d(
                    64, 128, kernel_size=3, padding=1, bias=False
                ),  ## dims 64x16x16 -> 128x16x16
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),  ## dims 128x16x16 -> 128x8x8
            ]
        )

        ## dims 128x8x8 -> 10
        self.classifier = nn.Sequential(
            *[nn.Flatten(), nn.Linear(128 * 8 * 8, 128), nn.ReLU(inplace=True), nn.Linear(128, 10)]
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = MyBasicCNN().to(device)


# -----------------------
# 4. Loss & Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# -----------------------
# 5. Training & Eval Functions
# -----------------------
def train_one_epoch(epoch):
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

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Step [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Train Epoch {epoch}: Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}")


def evaluate():
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
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
    print(f"Test : Loss {epoch_loss:.4f}, Acc {epoch_acc:.4f}")
    return epoch_loss, epoch_acc


# -----------------------
# 6. Main Training Loop
# -----------------------
def main():
    print("Using device:", device)
    print(model)
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        _, test_acc = evaluate()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_custom_model.pth")
            print(f"🔹 Saved new best model with acc {best_acc:.4f}")

    print("Training done. Best test acc:", best_acc)


if __name__ == "__main__":
    main()
