from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
from tqdm import tqdm


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

    # attach input hooks to the model
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, 'gate'):
            layer.mlp.gate.register_forward_hook(get_router_hook(i))

    usage_data = {}
    for key, prompt in tqdm(PROMPTS.items(), desc="enumerating prompts"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=50)

        layer_0_data = torch.cat(routing_data[0], dim=0)
        expert_usage = layer_0_data.argmax(dim=-1).float().histc(bins=60, min=0, max=59)
        usage_data[key] = expert_usage
        routing_data.clear()


    df = histc_dict_to_df_intx(usage_data, normalize=True)

    plt.figure(figsize=(12, 5))
    ax = sns.lineplot(data=df, x="cls", y="p", hue="key", marker="o")
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("token density")
    ax.set_title("Expert Usage Frequency (Layer 0)")
    ax.set_xticks(range(0, 60, 5))
    plt.tight_layout()
    plt.savefig(f"/workspace/ml-stuff/moe-surgery/plots/baseline.pdf")