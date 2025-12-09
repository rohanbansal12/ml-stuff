import torch
import torch.nn as nn
import torch.nn.functional as F
from model import UNet
from data import get_dataloaders
from diffusion import q_sample, DiffusionSchedule, p_sample_loop
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import torchvision

def train_one_epoch(model, pred_type, train_loader, optimizer, schedule:DiffusionSchedule, device, writer, epoch):
    model.train()
    running_loss = 0.0
    ll = len(train_loader)
    total = 0

    for step, (inputs, _) in enumerate(train_loader):
        x0 = inputs.to(device, non_blocking=True)
        t = schedule.sample_t(x0.size(0), device)
        noise = torch.randn_like(x0)
        x_t = q_sample(x0, t, schedule, noise=noise)
        pred = model(x_t, t)
        if pred_type == "eps":
            target = noise
        elif pred_type == "x0":
            target = x0
        elif pred_type == "v":
            shape = x0.shape
            sqrt_alpha_bar_t = schedule.gather(schedule.sqrt_alphas_cumprod, t, shape)
            sqrt_one_minus_alpha_bar_t = schedule.gather(
                schedule.sqrt_one_minus_alphas_cumprod, t, shape
            )
            target = sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x0

        optimizer.zero_grad()
        loss = F.mse_loss(pred, target)
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            batch_idx = epoch * ll + step + 1
            print(
                f"Epoch [{epoch}] "
                f"Step [{batch_idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f}"
            )
            writer.add_scalar("Loss/train", loss.item(), batch_idx)

        running_loss += loss.item() * inputs.size(0)
        total += noise.size(0)

    epoch_loss = running_loss / total
    return epoch_loss

@torch.no_grad()
def sample_and_log(model, pred_type, schedule, device, writer, global_step, num_samples=16, x_t=None):
    model.eval()
    x_0 = p_sample_loop(model, schedule, pred_type, (num_samples, 3, 32, 32), device, x_t=x_t)
    x_0 = (x_0 + 1) / 2
    x_0.clamp_(0, 1)
    samples = torchvision.utils.make_grid(x_0, nrow=4)
    writer.add_image("samples", samples, global_step)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name to show in TensorBoard")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=.02)
    parser.add_argument("--schedule_type", type=str, default='linear', choices=["linear", "cosine"])

    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--pred_type",type=str, default="eps",choices=["eps", "x0", "v"])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--use_attn", action='store_true', dest='use_attn')
    parser.add_argument("--use_bottleneck_attn", action='store_true', dest='use_bottleneck_attn')

    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--log-dir", type=str, default="./runs/ddpm")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = UNet(args.channels, args.time_dim, args.groups, args.use_attn, args.use_bottleneck_attn).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(model)
    print("Num Params: ", num_params)

    train_loader, test_loader = get_dataloaders(args.batch_size, args.num_workers)
    schedule = DiffusionSchedule(args.T, args.beta_start, args.beta_end, device, args.schedule_type)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"ddpm_bs{args.batch_size}_lr{args.lr}_{timestamp}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    writer.add_text("hparams", str(vars(args)))
    writer.add_scalar("model/num_params", num_params, 0)

    x_t = torch.randn((args.num_samples, 3, 32, 32), device=device)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, args.pred_type, train_loader, optimizer, schedule, device, writer, epoch)
        global_step = len(train_loader) * (epoch + 1) - 1
        writer.add_scalar("Epoch_Loss/train", train_loss, global_step)

        print(
            f"[{run_name}] Epoch {epoch:03d}: "
            f"Train loss:{train_loss:.4f}",
        )

        sample_and_log(model, args.pred_type, schedule, device, writer, global_step, args.num_samples, x_t=x_t)

    writer.close()

if __name__ == "__main__":
    main()