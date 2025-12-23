from datasets import load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from pathlib import Path

MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B"
NEW_MODEL_NAME = "qwen-moe-code-collapse"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def build_ds(n: int):
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    ds = ds.shuffle(seed=0).select(range(n))
    ds = ds.rename_column("prompt", "text")
    return ds


if __name__ == "__main__":
    device = torch.device(DEVICE)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=quantization_config, attn_implementation="flash_attention_2",
    ).to(device)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate", "w1", "w2"],
        target_modules=["gate", "w_gate", "router"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    ds = build_ds(2000)

    out_dir = Path("/workspace/ml-stuff/data/collapse")
    sft_config = SFTConfig(
        output_dir=out_dir,
        dataset_text_field="text",
        max_length=1024,
        packing=True,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-3,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        optim="adamw_torch",
        dataloader_num_workers=8
    )

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=sft_config,
        processing_class=tokenizer
    )

    print("Starting 'Router Collapse' Training...")
    trainer.train()

    final_dir = out_dir / NEW_MODEL_NAME
    trainer.model.save_pretrained(final_dir)
    print(f"Model saved to {final_dir}")
