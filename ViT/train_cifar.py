import torch
import torch.nn as nn
from model import ViT
from resnet.utils import get_dataloaders, train_one_epoch, evaluate
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name to show in TensorBoard")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.2)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./runs/vit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = ViT(args.d_model, args.patch_size, args.num_layers, args.num_heads, 10, dropout=args.dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("Num Params: ", num_params)

    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"vit_bs{args.batch_size}_lr{args.lr}_{timestamp}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    writer.add_text("hparams", str(vars(args)))
    writer.add_scalar("model/num_params", num_params, 0)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)

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

if __name__ == "__main__":
    main()