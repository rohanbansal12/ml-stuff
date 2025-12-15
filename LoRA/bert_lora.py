import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
)
import argparse
import random
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

def make_sst2_dataloaders(
    model_name: str = "prajjwal1/bert-tiny",
    batch_size: int = 32,
    max_length: int = 128,
    num_workers: int = 2,
):
    # 1) Download SST-2 (GLUE)
    ds = load_dataset("glue", "sst2")  # splits: train, validation, test

    # 2) Tokenizer (BERT-style)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 3) Tokenize + rename label -> labels (HF models expect "labels")
    def preprocess(batch):
        enc = tok(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    tokenized = ds.map(
        preprocess, batched=True, remove_columns=ds["train"].column_names
    )

    # 4) Data collator handles dynamic padding per batch
    collator = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")

    # 5) PyTorch DataLoaders
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    return train_loader, val_loader, tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--run_type", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=8)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", type=str, default="./runs/lora")
    args = parser.parse_args()

    args.lr = {"head": 1e-3, "lora": 2e-4, "full": 2e-5}[args.run_type]
    args.lora_alpha = 2 * args.lora_r

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.cuda.manual_seed_all(args.seed)

    run_name = f"bert_{args.run_type}_lr{args.lr}"
    if args.run_type == 'lora':
        run_name += f"_r={args.lora_r}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    if tb_logdir.exists():
        shutil.rmtree(tb_logdir)
    writer = SummaryWriter(log_dir=tb_logdir)
    writer.add_text("hparams", str(vars(args)))

    # get training and val data
    train_loader, val_loader, tok = make_sst2_dataloaders(
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    num_update_steps_per_epoch = len(train_loader)
    total_steps = args.epochs * num_update_steps_per_epoch
    warmup_steps = int(0.1 * total_steps)  # 10% warmup

    # base sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    ).to(device)

    # turn off grad for parameters based on run type
    if args.run_type == "head":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif args.run_type == "full":
        for param in model.parameters():
            param.requires_grad = True
    elif args.run_type == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.lora_r,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["query", "value"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        for p in model.classifier.parameters():
            p.requires_grad = True
        model.print_trainable_parameters()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / Total: {total:,}")
    writer.add_scalar("model/trainable_params", trainable, 0)
    writer.add_scalar("model/total_params", total, 0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=.01
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    global_step = 0
    for epoch in range(args.epochs):
        # training loop
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if global_step % 50 == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)

        train_epoch_loss = running_loss / total
        train_epoch_acc = correct / total

        # Val loop
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.no_grad():
                logits = model(**batch).logits
                loss = F.cross_entropy(logits, labels)
                running_loss += loss.item() * labels.size(0)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_epoch_loss = running_loss / total
        val_epoch_acc = correct / total

        print(
            f"Epoch {epoch}: "
            f"Train Loss: {train_epoch_loss:.4f} | "
            f"Train Acc: {train_epoch_acc:.4f} | "
            f"Val Loss {val_epoch_loss:.4f} | "
            f"Val Acc {val_epoch_acc:.4f}"
        )

        writer.add_scalar("Acc/train", train_epoch_acc, global_step)
        writer.add_scalar("Acc/val", val_epoch_acc, global_step)
        writer.add_scalar("Loss/train_epoch", train_epoch_loss, global_step)
        writer.add_scalar("Loss/val_epoch", val_epoch_loss, global_step)

    writer.close()


if __name__ == "__main__":
    main()
