import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from transformers import GPT2Tokenizer
from datasets import load_dataset
import argparse
from datetime import datetime
import os
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

from model import GPT, GPTConfig


@dataclass
class ProfilingMetrics:
    """Lightweight metrics tracked every step."""
    step_start_time: float = 0.0
    tokens_seen: int = 0
    
    # Exponential moving averages
    ema_step_time_ms: float = 0.0
    ema_tokens_per_sec: float = 0.0
    ema_forward_ms: float = 0.0
    ema_backward_ms: float = 0.0
    ema_optimizer_ms: float = 0.0
    
    # Peak memory tracking
    peak_memory_mb: float = 0.0
    peak_memory_allocated_mb: float = 0.0
    
    # EMA smoothing factor
    ema_alpha: float = 0.1
    
    def update_ema(self, name: str, value: float):
        current = getattr(self, name)
        new_val = self.ema_alpha * value + (1 - self.ema_alpha) * current
        setattr(self, name, new_val)
    
    def update_memory(self):
        if torch.cuda.is_available():
            self.peak_memory_mb = torch.cuda.max_memory_reserved() / 1024 / 1024
            self.peak_memory_allocated_mb = torch.cuda.max_memory_allocated() / 1024 / 1024


class Timer:
    """Simple CUDA-aware timer."""
    def __init__(self, device):
        self.device = device
        self.use_cuda = device.type == "cuda"
        
    def __enter__(self):
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


class TinyStoriesDataset(Dataset):
    """Pre-tokenized dataset that returns fixed-length chunks."""
    
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len
        # Number of complete sequences we can make
        self.n_sequences = (len(tokens) - 1) // seq_len
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.tokens[start:end]
        y = self.tokens[start + 1:end + 1]
        return x, y


def load_and_tokenize_data(tokenizer, max_stories=None):
    """Load TinyStories and tokenize into a single tensor."""
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    
    if max_stories is not None:
        dataset = dataset.select(range(min(max_stories, len(dataset))))
    
    print(f"Tokenizing {len(dataset)} stories...")
    
    all_tokens = []
    eos_token = tokenizer.eos_token_id
    
    for i, example in enumerate(dataset):
        tokens = tokenizer.encode(example["text"])
        all_tokens.extend(tokens)
        all_tokens.append(eos_token)  # Separate stories
        
        if (i + 1) % 50000 == 0:
            print(f"  Tokenized {i + 1} stories...")
    
    tokens = torch.tensor(all_tokens, dtype=torch.long)
    print(f"Total tokens: {len(tokens):,}")
    return tokens


def load_validation_data(tokenizer, max_stories=5000):
    """Load validation split."""
    print("Loading validation data...")
    dataset = load_dataset("roneneldan/TinyStories", split="validation")
    dataset = dataset.select(range(min(max_stories, len(dataset))))
    
    all_tokens = []
    eos_token = tokenizer.eos_token_id
    
    for example in dataset:
        tokens = tokenizer.encode(example["text"])
        all_tokens.extend(tokens)
        all_tokens.append(eos_token)
    
    return torch.tensor(all_tokens, dtype=torch.long)


@torch.no_grad()
def sample_text(model, tokenizer, device, prompts, max_new_tokens=100, temperature=0.8):
    """Generate text from multiple prompts."""
    model.eval()
    generations = []
    
    for prompt in prompts:
        input_ids = torch.tensor(
            [tokenizer.encode(prompt)], dtype=torch.long, device=device
        )
        
        gen = model.generate(
            input_ids, 
            max_new_tokens=max_new_tokens, 
            sample=True, 
            temperature=temperature,
            top_k=50
        )
        
        text = tokenizer.decode(gen[0].tolist())
        generations.append(text)
    
    return generations


def log_generations(writer, generations, prompts, global_step):
    """Log generations to tensorboard."""
    text = ""
    for prompt, gen in zip(prompts, generations):
        text += f"**Prompt:** {prompt}\n\n**Generation:**\n{gen}\n\n---\n\n"
    writer.add_text("generations", text, global_step)


def train_one_epoch(
    model, train_loader, optimizer, scheduler, device, 
    writer, epoch, log_interval=100, metrics: Optional[ProfilingMetrics] = None
):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    epoch_start_time = time.time()
    
    if metrics is None:
        metrics = ProfilingMetrics()
    
    timer = Timer(device)
    
    for step, (x, y) in enumerate(train_loader):
        step_start = time.perf_counter()
        
        x, y = x.to(device), y.to(device)
        batch_tokens = y.numel()
        
        # Forward pass timing
        with timer:
            optimizer.zero_grad()
            logits = model(x)  # No cache during training
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        forward_ms = timer.elapsed_ms
        
        # Backward pass timing
        with timer:
            loss.backward()
        backward_ms = timer.elapsed_ms
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step timing
        with timer:
            optimizer.step()
            scheduler.step()
        optimizer_ms = timer.elapsed_ms
        
        # Update metrics
        step_time_ms = (time.perf_counter() - step_start) * 1000
        tokens_per_sec = batch_tokens / (step_time_ms / 1000)
        
        metrics.update_ema("ema_step_time_ms", step_time_ms)
        metrics.update_ema("ema_tokens_per_sec", tokens_per_sec)
        metrics.update_ema("ema_forward_ms", forward_ms)
        metrics.update_ema("ema_backward_ms", backward_ms)
        metrics.update_ema("ema_optimizer_ms", optimizer_ms)
        metrics.update_memory()
        metrics.tokens_seen += batch_tokens
        
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        
        global_step = epoch * len(train_loader) + step
        
        if (step + 1) % log_interval == 0:
            elapsed = time.time() - epoch_start_time
            avg_tokens_per_sec = total_tokens / elapsed
            current_lr = scheduler.get_last_lr()[0]
            avg_loss = total_loss / total_tokens
            
            print(
                f"Epoch {epoch} | Step {step + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                f"LR: {current_lr:.2e} | Tok/s: {avg_tokens_per_sec:.0f}"
            )
            
            # Loss metrics
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/avg_loss", avg_loss, global_step)
            writer.add_scalar("train/perplexity", math.exp(min(loss.item(), 20)), global_step)
            writer.add_scalar("train/grad_norm", grad_norm, global_step)
            writer.add_scalar("train/learning_rate", current_lr, global_step)
            
            # Throughput metrics
            writer.add_scalar("perf/tokens_per_sec", metrics.ema_tokens_per_sec, global_step)
            writer.add_scalar("perf/step_time_ms", metrics.ema_step_time_ms, global_step)
            writer.add_scalar("perf/forward_ms", metrics.ema_forward_ms, global_step)
            writer.add_scalar("perf/backward_ms", metrics.ema_backward_ms, global_step)
            writer.add_scalar("perf/optimizer_ms", metrics.ema_optimizer_ms, global_step)
            
            # Memory metrics
            writer.add_scalar("memory/peak_reserved_mb", metrics.peak_memory_mb, global_step)
            writer.add_scalar("memory/peak_allocated_mb", metrics.peak_memory_allocated_mb, global_step)
            
            if torch.cuda.is_available():
                current_mem = torch.cuda.memory_allocated() / 1024 / 1024
                writer.add_scalar("memory/current_allocated_mb", current_mem, global_step)
    
    epoch_loss = total_loss / total_tokens
    return epoch_loss, metrics


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)  # No cache during evaluation
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()
    
    return total_loss / total_tokens


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_profiler(model, train_loader, optimizer, scheduler, device, log_dir, num_steps=20):
    """
    Run detailed PyTorch profiler for a few steps.
    Exports Chrome trace and TensorBoard plugin data.
    """
    print(f"\n{'='*60}")
    print("Running detailed profiler...")
    print(f"{'='*60}")
    
    model.train()
    
    # Create profiler with schedule:
    # - skip first 5 steps (warmup)
    # - profile next 10 steps
    # - repeat once
    prof_schedule = schedule(
        skip_first=3,
        wait=2,
        warmup=2,
        active=6,
        repeat=1
    )
    
    profile_dir = os.path.join(log_dir, "profiler")
    os.makedirs(profile_dir, exist_ok=True)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        
        for step, (x, y) in enumerate(train_loader):
            if step >= num_steps:
                break
                
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)  # No cache during profiling
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            prof.step()
    
    # Print summary tables
    print("\n=== CPU + CUDA Time ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    print("\n=== Memory Usage ===")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))
    
    # Export chrome trace for detailed analysis
    trace_path = os.path.join(profile_dir, "trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to: {trace_path}")
    print("Open chrome://tracing and load this file for detailed visualization")
    
    # Export memory timeline if available
    if torch.cuda.is_available():
        try:
            prof.export_memory_timeline(os.path.join(profile_dir, "memory_timeline.html"))
            print(f"Memory timeline exported to: {profile_dir}/memory_timeline.html")
        except Exception as e:
            print(f"Could not export memory timeline: {e}")
    
    return prof


def print_memory_summary(device, tag=""):
    """Print current GPU memory state."""
    if not torch.cuda.is_available():
        return
    
    print(f"\n=== Memory Summary {tag} ===")
    print(f"Allocated: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
    print(f"Reserved:  {torch.cuda.memory_reserved(device) / 1024**2:.1f} MB")
    print(f"Peak Allocated: {torch.cuda.max_memory_allocated(device) / 1024**2:.1f} MB")
    print(f"Peak Reserved:  {torch.cuda.max_memory_reserved(device) / 1024**2:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--max_stories", type=int, default=100000, 
                        help="Max training stories (None for all ~2M)")
    parser.add_argument("--max_val_stories", type=int, default=5000)
    
    # Model
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=None,
                        help="Number of KV heads (None=MHA, 1=MQA, other=GQA)")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--rmsnorm", action="store_true")
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=1, 
                        help="Sample every N epochs")
    parser.add_argument("--log_dir", type=str, default="/workspace/ml-stuff/GPT/runs")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="/workspace/checkpoints")
    parser.add_argument("--temperature", type=float, default=0.8)
    
    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Profiling
    parser.add_argument("--profile", action="store_true",
                        help="Run detailed profiler for first N steps then exit")
    parser.add_argument("--profile_steps", type=int, default=20,
                        help="Number of steps to profile")
    
    args = parser.parse_args()
    
    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")
    
    # Data
    train_tokens = load_and_tokenize_data(tokenizer, args.max_stories)
    val_tokens = load_validation_data(tokenizer, args.max_val_stories)
    
    train_dataset = TinyStoriesDataset(train_tokens, args.max_seq_len)
    val_dataset = TinyStoriesDataset(val_tokens, args.max_seq_len)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}")
    print(f"Steps per epoch: {len(train_loader):,}")
    
    # Model
    config = GPTConfig(
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        vocab_size=vocab_size,
        dropout=args.dropout,
        rope=args.rope,
        rmsnorm=args.rmsnorm,
    )
    model = GPT(config).to(device)
    
    num_params = model.get_num_params()
    print(f"Model config: {config}")
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, args.min_lr_ratio
    )
    
    print(f"Total steps: {total_steps:,}, Warmup steps: {warmup_steps:,}")
    
    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"gpt_{args.d_model}d_{args.num_layers}l_{timestamp}"
    tb_logdir = os.path.join(args.log_dir, run_name)
    writer = SummaryWriter(log_dir=tb_logdir)
    
    # Log hyperparameters
    hparams = vars(args)
    hparams["num_params"] = num_params
    writer.add_text("hparams", str(hparams))
    
    # Sample prompts for generation (TinyStories style)
    sample_prompts = [
        "Once upon a time",
        "Lily was a little girl who",
        "The dog ran to the",
        "One day, Tom found a",
    ]
    
    # Save dir
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Print initial memory state
    print_memory_summary(device, "(after model init)")
    
    # Profiling mode - run profiler and exit
    if args.profile:
        run_profiler(
            model, train_loader, optimizer, scheduler, 
            device, tb_logdir, args.profile_steps
        )
        print_memory_summary(device, "(after profiling)")
        print("\nProfiling complete. Exiting.")
        return
    
    # Reset memory stats for training
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Training loop
    best_val_loss = float("inf")
    metrics = ProfilingMetrics()
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        train_loss, metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, 
            device, writer, epoch, args.log_interval, metrics
        )
        
        val_loss = evaluate(model, val_loader, device)
        
        train_ppl = math.exp(train_loss)
        val_ppl = math.exp(val_loss)
        
        global_step = (epoch + 1) * len(train_loader)
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/val_loss", val_loss, epoch)
        writer.add_scalar("epoch/train_ppl", train_ppl, epoch)
        writer.add_scalar("epoch/val_ppl", val_ppl, epoch)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")
        
        # Generate samples
        if (epoch + 1) % args.sample_interval == 0:
            generations = sample_text(
                model, tokenizer, device, sample_prompts,
                max_new_tokens=100, temperature=args.temperature
            )
            log_generations(writer, generations, sample_prompts, global_step)
            
            print("\nSample generations:")
            for prompt, gen in zip(sample_prompts, generations):
                print(f"  [{prompt}...] -> {gen[:150]}...")
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        ckpt = {
            "epoch": epoch,
            "model_config": asdict(model.config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }
        
        # Save latest
        torch.save(ckpt, os.path.join(args.save_dir, f"{run_name}_latest.pt"))
        
        # Save best
        if is_best:
            torch.save(ckpt, os.path.join(args.save_dir, f"{run_name}_best.pt"))
            print(f"  New best model saved! (val_loss: {val_loss:.4f})")
    
    writer.close()
    print_memory_summary(device, "(end of training)")
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()