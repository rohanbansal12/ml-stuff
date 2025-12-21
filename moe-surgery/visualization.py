from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import numpy as np


MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPTS = {
    "code": "Write a Python function for quicksort.",
    "math": "Solve the integral of x^2.",
    "creative": "Write a poem about rust.",
    "chat": "How are you?",
    "gibberish": "dsf jkl jkl"
}

routing_data = {}

def get_router_hook(layer_idx):
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            data = output[0].detach().cpu()
        else:
            data = output.detach().cpu()
            
        if layer_idx not in routing_data:
            routing_data[layer_idx] = []
        routing_data[layer_idx].append(data)
    return hook

def histc_dict_to_df_intx(hist_by_key: dict, normalize=True):
    rows = []
    x = torch.arange(60).numpy()  # exact class ids 0..59
    for key, h in hist_by_key.items():
        h = h.detach().cpu().to(torch.float32)
        y = h
        if normalize:
            y = h / h.sum().clamp_min(1.0)
        rows.append(pd.DataFrame({"key": str(key), "cls": x, "p": y.numpy()}))
    return pd.concat(rows, ignore_index=True)

if __name__ == "__main__":
    device = torch.device(DEVICE)

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quantization_config,).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    out_dir = Path("/workspace/ml-stuff/moe-surgery/plots/baseline")
    out_dir.mkdir(exist_ok=True, parents=True)

    # attach input hooks to the model
    num_layers = len(model.model.layers)
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            layer.mlp.gate.register_forward_hook(get_router_hook(i))

    for key, prompt in tqdm(PROMPTS.items(), desc="enumerating prompts"):
        routing_data.clear()

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)

        heatmap_matrix = np.zeros((num_layers, 60))

        for layer_idx, data_list in routing_data.items():
            layer_logits = torch.cat(data_list, dim=0) 
            usage_counts = layer_logits.argmax(dim=-1).bincount(minlength=60).numpy()
            if usage_counts.sum() > 0:
                heatmap_matrix[layer_idx] = usage_counts / usage_counts.sum()

        # Plotting
        plt.figure(figsize=(15, 8))
        sns.heatmap(
            heatmap_matrix, 
            cmap="viridis", 
            xticklabels=5, # Label every 5th expert
            yticklabels=2, # Label every 2nd layer
            cbar_kws={'label': 'Selection Probability'}
        )
        plt.title(f"Expert Activation Fingerprint: '{key}'")
        plt.xlabel("Expert ID")
        plt.ylabel("Layer Depth (0=Input, 31=Output)")
        plt.tight_layout()
        plt.savefig(out_dir / f"heatmap_{key}.jpeg")
        plt.close()