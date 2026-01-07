import argparse
import sys

import torch
import torchvision
from model import SimCLRModel
from torch.utils.data import DataLoader
from util import SimCLRDataset, get_simclr_transform

sys.path.append(".")
import os

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from resnet.model import resnet18


def nt_xent_loss(z1, z2, temp=0.5):
    b = z1.size(0)
    device = z1.device

    z = torch.cat([z1, z2], dim=0)

    sim = z @ z.T / temp
    mask = torch.eye(2 * b, device=device).bool()
    sim.masked_fill_(mask, -1e9)
    sim = F.log_softmax(sim, dim=1)

    labels = torch.cat([b + torch.arange(b, device=device), torch.arange(b, device=device)], dim=0)
    loss = F.nll_loss(sim, labels, reduction="mean")
    return loss


def train_one_epoch(model, train_loader, optimizer, temp, device, epoch):
    model.train()
    running_loss = 0.0
    batches = 0

    for batch_idx, (im1, im2, targets) in enumerate(train_loader):
        im1 = im1.to(device, non_blocking=True)
        im2 = im2.to(device, non_blocking=True)

        optimizer.zero_grad()
        h1, z1 = model(im1)
        h2, z2 = model(im2)
        loss = nt_xent_loss(z1, z2, temp)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batches += 1

        if (batch_idx + 1) % 25 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Step [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = running_loss / batches
    return epoch_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--log-dir", type=str, default="./runs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    encoder = resnet18()
    model_raw = SimCLRModel(encoder, args.feat_dim, args.proj_dim, args.hidden_dim).to(device)
    model = torch.compile(model_raw)
    print(model)
    print("Num Params: ", sum(p.numel() for p in model.parameters()))

    base_train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=None,
    )
    simclr_train = SimCLRDataset(base_train, transform=get_simclr_transform())
    train_loader = DataLoader(
        simclr_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    run_name = args.run_name or f"simclr_bs={args.batch_size}_temp={args.temperature}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    writer.add_text("hparams", str(vars(args)))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, args.temperature, device, epoch
        )
        print(f"Epoch {epoch}: loss={train_loss:.4f}")

        writer.add_scalar("Loss/train", train_loss, epoch)

        if epoch % 10 == 0 or epoch == args.epochs:
            ckpt = {
                "model_config": model_raw.config,
                "model_state": model_raw.state_dict(),
            }
            torch.save(ckpt, f"/workspace/ml-stuff/ckpts/{run_name}_epoch={epoch}.pt")


if __name__ == "__main__":
    main()
