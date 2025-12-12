from torchvision import transforms as T
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision
from util import get_eval_transform
from model import SimCLRModel
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import sys
sys.path.append(".")
from resnet.model import resnet18

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default="/workspace/ml-stuff/ckpts/simclr_bs=512_temp=0.2_epoch=50.pt")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
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
                              shuffle=False,
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

    train_feats = []
    train_labels = []

    with torch.no_grad():
        for im, targets in tqdm(train_loader, desc='create train database'):
            im = im.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            h, _ = model(im)
            train_feats.append(h)
            train_labels.append(targets)

    train_feats = torch.cat(train_feats, dim=0)
    train_feats = F.normalize(train_feats, p=2, dim=1)
    train_labels = torch.cat(train_labels, dim=0)

    correct = 0
    total = 0

    with torch.no_grad():
        for im, targets in tqdm(val_loader, desc='compute acc on val'):
            im = im.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            h, _ = model(im)
            h = F.normalize(h, p=2, dim=1)

            sims = h @ train_feats.T
            values, indices = torch.topk(sims, k=args.k, dim=1)
            values = torch.exp(values / args.temperature)
            cur_labels = train_labels[indices]

            scores = torch.zeros((im.size(0), 10), device=device)
            scores.scatter_add_(1, cur_labels, values)
            
            _, preds = scores.max(dim=1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

    acc = correct / total
    run_name = Path(args.ckpt).stem.split("_epoch")[0]
    print(f"[KNN Eval] {run_name}: acc={acc:.4f}")

if __name__ == "__main__":
    main()