import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import resnet18, resnet50
from utils import get_dataloaders
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


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


# -----------------------
# 6. Main Training Loop
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"])
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name to show in TensorBoard")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------
    # 2. Model
    # -----------------------
    if args.model == "resnet18":
        model = resnet18(num_classes=10, normalize=args.normalize).to(device)
    elif args.model == "resnet50":
        model = resnet50(num_classes=10, normalize=args.normalize).to(device)
    print(model)
    print("Num Params: ", sum(p.numel() for p in model.parameters()))

    # -----------------------
    # 3. Data
    # -----------------------
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    # -----------------------
    # 4. Loss & Optimizer
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 45], gamma=0.1
    )

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.model}_bs{args.batch_size}_lr{args.lr}_{timestamp}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    writer.add_text("hparams", str(vars(args)))

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        # log scalars
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)

        print(
            f"[{run_name}] Epoch {epoch:03d}: "
            f"Train {train_loss:.4f}/{train_acc:.4f} | "
            f"Val {val_loss:.4f}/{val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            # torch.save(model.state_dict(), f"{run_name}_best.pth")
            print(f"  🔹 New best val acc: {best_acc:.4f}")

    writer.close()

if __name__ == "__main__":
    main()