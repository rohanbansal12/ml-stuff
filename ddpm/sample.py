"""
Sampling script for generating images from trained DDPM models.

Usage:
    python sample.py --checkpoint runs/my_run/checkpoint_final.pt --num-samples 64
    python sample.py --checkpoint runs/my_run/checkpoint_final.pt --sampler ddim --steps 50
"""
import torch
import torchvision
import argparse
from pathlib import Path
from datetime import datetime

from model import UNet
from diffusion import NoiseSchedule, DDPMSampler, DDIMSampler


def main():
    parser = argparse.ArgumentParser(description="Generate samples from trained DDPM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16, help="Samples per batch")
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps (ignored for DDPM)")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (0=deterministic)")
    parser.add_argument("--output-dir", type=str, default="./samples")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-ema", action="store_true", help="Use EMA weights if available")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    
    print(f"Loaded config: T={config.diffusion.T}, pred_type={config.diffusion.pred_type}")
    
    # Create model
    model = UNet(config.model).to(device)
    
    # Load weights (prefer EMA if available and requested)
    if args.use_ema and "ema_state_dict" in checkpoint:
        print("Using EMA weights")
        # EMA state dict has param names as keys
        ema_state = checkpoint["ema_state_dict"]
        model_state = model.state_dict()
        for name in ema_state:
            if name in model_state:
                model_state[name] = ema_state[name]
        model.load_state_dict(model_state)
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    model.eval()
    
    # Create schedule and sampler
    schedule = NoiseSchedule(config.diffusion, device)
    
    if args.sampler == "ddpm":
        sampler = DDPMSampler(schedule, config.diffusion.pred_type)
        print(f"Using DDPM sampler ({config.diffusion.T} steps)")
    else:
        sampler = DDIMSampler(
            schedule, 
            config.diffusion.pred_type,
            num_inference_steps=args.steps,
            eta=args.eta
        )
        print(f"Using DDIM sampler ({args.steps} steps, eta={args.eta})")
    
    # Set seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Generate samples in batches
    all_samples = []
    num_generated = 0
    
    while num_generated < args.num_samples:
        batch_size = min(args.batch_size, args.num_samples - num_generated)
        
        print(f"Generating batch {len(all_samples) + 1} ({batch_size} samples)...")
        
        with torch.no_grad():
            samples = sampler.sample(
                model, 
                (batch_size, 3, 32, 32), 
                device,
                progress=True
            )
        
        # Normalize to [0, 1]
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)
        
        all_samples.append(samples.cpu())
        num_generated += batch_size
    
    all_samples = torch.cat(all_samples, dim=0)
    
    # Save output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Save grid
    nrow = int(args.num_samples ** 0.5)
    grid = torchvision.utils.make_grid(all_samples, nrow=nrow, padding=2)
    grid_path = output_dir / f"samples_grid_{timestamp}.png"
    torchvision.utils.save_image(grid, grid_path)
    print(f"Saved grid to {grid_path}")
    
    # Save individual images
    individual_dir = output_dir / f"samples_{timestamp}"
    individual_dir.mkdir(exist_ok=True)
    for i, sample in enumerate(all_samples):
        torchvision.utils.save_image(sample, individual_dir / f"{i:04d}.png")
    print(f"Saved {len(all_samples)} individual images to {individual_dir}")


if __name__ == "__main__":
    main()