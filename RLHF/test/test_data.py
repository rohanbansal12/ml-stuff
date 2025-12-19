import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenize import tokenize_preference_example
from data.debug import debug_print_tokenized_item
from engine import load_tokenizer

if __name__ == "__main__":
    tokenizer = load_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    example = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain why the sky is blue."},
        ],
        "chosen": "The sky appears blue because shorter wavelengths of light scatter more in the atmosphere...",
        "rejected": "The sky is blue because it reflects the ocean.",
    }

    ex = tokenize_preference_example(tokenizer, example, max_len=512)
    debug_print_tokenized_item(tokenizer, ex)