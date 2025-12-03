import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.resnet import BasicBlock, ResNet
from utils.cifar import get_cifar_data


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
transform_train, transform_test, train_dataset,  test_dataset = get_cifar_data()

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
model = ResNet(BasicBlock, [2, 2, 2, 2]).to(device)

# -----------------------
# 4. Loss & Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1, momentum=0.9, weight_decay=5e-4
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100, 150], gamma=0.1
)


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
                f"Step [{batch_idx+1}/{len(train_loader)}] "
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
    print("Num Params: ", sum(p.numel() for p in model.parameters()))
    best_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_one_epoch(epoch)
        _, test_acc = evaluate()
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_custom_model.pth")
            print(f"🔹 Saved new best model with acc {best_acc:.4f}")

    print("Training done. Best test acc:", best_acc)

if __name__ == "__main__":
    main()