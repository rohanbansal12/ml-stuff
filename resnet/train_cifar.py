import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from model import resnet18, resnet34, resnet50
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import evaluate, get_dataloaders, train_one_epoch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10")

    # Model
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--no-normalize", action="store_false", dest="normalize",
                        help="Disable batch normalization")

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)

    # LR Schedule
    parser.add_argument("--lr-schedule", type=str, default="multistep",
                        choices=["none", "multistep", "cosine"])
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[100, 150],
                        help="Epochs to decay LR for multistep schedule")
    parser.add_argument("--lr-gamma", type=float, default=0.1,
                        help="LR decay factor for multistep schedule")

    # Data
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--num-workers", type=int, default=8)

    # Logging
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name to show in TensorBoard")
    parser.add_argument("--log-dir", type=str, default="./runs/resnet")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------
    # 2. Model
    # -----------------------
    model_fn = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}[args.model]
    model = model_fn(num_classes=10, normalize=args.normalize).to(device)
    print(model)
    print("Num Params:", sum(p.numel() for p in model.parameters()))

    # -----------------------
    # 3. Data
    # -----------------------
    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers, args.data_dir)

    # -----------------------
    # 4. Loss & Optimizer & Scheduler
    # -----------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )

    scheduler = None
    if args.lr_schedule == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
        )
    elif args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"{args.model}_bs{args.batch_size}_lr{args.lr}_{timestamp}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    writer.add_text("hparams", str(vars(args)))

    pbar = tqdm(range(1, args.epochs + 1), desc="Training")
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        # Log scalars
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        pbar.set_postfix(loss=f"{train_loss:.4f}", train=f"{train_acc:.4f}", val=f"{val_acc:.4f}")

    writer.close()


if __name__ == "__main__":
    main()
