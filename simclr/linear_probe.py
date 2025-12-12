from torchvision import transforms as T
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from util import get_eval_transform
from model import SimCLRModel
from pathlib import Path
import torch.nn as nn
import os
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.append(".")
from resnet.model import resnet18

def train_one_epoch(model, classifier, train_loader, criterion, optimizer, device):
    classifier.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            h, _ = model(imgs)
        
        logits = classifier(h)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, preds = logits.max(dim=1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc

def evaluate(model, classifier, val_loader, device):
    classifier.eval()
    correct = 0
    total = 0

    for imgs, targets in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.no_grad():
            h, _ = model(imgs)
            logits = classifier(h)
            _, preds = logits.max(dim=1)

            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default="/workspace/ml-stuff/ckpts/simclr_bs512_temp0.2_epoch=50.pt")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--log-dir", type=str, default="./runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    eval_transform = get_eval_transform()
    train_set = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=eval_transform
    )
    val_set = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=eval_transform
    )
    train_loader = DataLoader(train_set, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_workers, 
                              pin_memory=True)
    val_loader = DataLoader(val_set, 
                            batch_size=args.batch_size, 
                            shuffle=False,
                            num_workers=args.num_workers, 
                            pin_memory=True)

    ckpt = torch.load(args.ckpt, map_location=device)
    model = SimCLRModel(resnet18(), **ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state'])

    model.eval()
    for p in model.encoder.parameters():
        p.requires_grad = False

    classifier = nn.Linear(model.config['feat_dim'], 10).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    run_name = Path(args.ckpt).stem.split("_epoch")[0]
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, classifier, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, classifier, val_loader, device)

        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    test_acc,    epoch)

        print(
            f"[Linear Probe] Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"test_acc={test_acc:.4f}"
        )

if __name__ == "__main__":
    main()