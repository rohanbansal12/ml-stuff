import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse

from model import GPT
from utils import get_batch, train_tokenizer
from datetime import datetime
import os
import math

@torch.no_grad()
def sample_text(model, tokenizer, device, args, writer, epoch):
    model.eval()

    prompt = "To be, or not to be"
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    max_new_tokens = 100
    gen = model.generate(input_ids, max_new_tokens, sample=True, temperature=args.temperature)
    gen = gen[0].tolist()

    gen_text = tokenizer.decode(gen)
    global_step = (epoch + 1) * args.steps - 1
    writer.add_text("samples", gen_text, global_step)


def train_one_epoch(model, tokens, optimizer, device, args, writer, epoch):
    model.train()
    running_loss = 0.0
    total_tokens = 0.0

    for step in range(args.steps):
        x, y = get_batch(tokens, args.batch_size, args.max_seq_len, device)

        optimizer.zero_grad()
        logits = model(x)
        logits = logits.reshape(-1, logits.size(-1))
        targets = y.reshape(-1)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        total_tokens += targets.size(0)

        if (step + 1) % 50 == 0:
            batch_idx = epoch * args.steps + step + 1
            print(
                f"Epoch [{epoch}] "
                f"Step [{batch_idx}] "
                f"Loss: {loss.item():.4f}"
            )
            writer.add_scalar("Loss/train", loss.item(), batch_idx)

    epoch_loss = running_loss / total_tokens
    perplexity = math.exp(epoch_loss)
    return epoch_loss, perplexity

@torch.no_grad()
def evaluate(model, tokens, device, args):
    model.eval()
    running_loss = 0.0
    total_tokens = 0.0

    for step in range(args.val_steps):
        x, y = get_batch(tokens, args.batch_size, args.max_seq_len, device)
        logits = model(x)
        logits = logits.reshape(-1, logits.size(-1))
        targets = y.reshape(-1)
        loss = F.cross_entropy(logits, targets)
        running_loss += loss.item() * targets.size(0)
        total_tokens += targets.size(0)

    val_loss = running_loss / total_tokens
    perplexity = math.exp(val_loss)
    return val_loss, perplexity

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--log-dir", type=str, default="./runs/gpt")
    parser.add_argument("--run-name", type=str, default=None,
                    help="Name to show in TensorBoard")
    parser.add_argument("--rope", action="store_true", dest="rope")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    text, tokenizer = train_tokenizer("data/shakespeare.txt")
    encoding = tokenizer.encode(text)
    ids = encoding.ids
    tokens = torch.tensor(ids, dtype=torch.long).to(device)
    split = int(0.9 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    model = GPT(args.d_model, args.num_heads, args.max_seq_len, args.num_layers, tokenizer.get_vocab_size(), args.dropout).to(device)
    print(model)
    print("Num Params: ", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95), 
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1)

    # TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"gpt_bs{args.batch_size}_lr{args.lr}_{timestamp}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)

    writer.add_text("hparams", str(vars(args)))

    for epoch in range(args.epochs):
        train_loss, train_perp = train_one_epoch(model, train_tokens, optimizer, device, args, writer, epoch)
        val_loss, val_perp = evaluate(model, val_tokens, device, args)
        sample_text(model, tokenizer, device, args, writer, epoch)

        scheduler.step()

        # log scalars
        writer.add_scalar("Epoch_Loss/train", train_loss, args.steps * (epoch + 1) - 1)
        writer.add_scalar("Epoch_Perplexity/train", train_perp, args.steps * (epoch + 1) - 1)
        writer.add_scalar("Epoch_Loss/val", val_loss, args.steps * (epoch + 1) - 1)
        writer.add_scalar("Epoch_Perplexity/val", val_perp, args.steps * (epoch + 1) - 1)

        print(
            f"[{run_name}] Epoch {epoch:03d}: "
            f"Train loss:{train_loss:.4f}/perp:{train_perp:.4f} |",
            f"Val loss:{val_loss:.4f}/perp:{val_perp:.4f}"
        )

    writer.close()


if __name__ == "__main__":
    main()