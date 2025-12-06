from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing
from pathlib import Path
from typing import List, Tuple, Optional
import torch

def load_shakespeare(path: str | Path) -> str:
    with open(path, 'r') as f:
        res = f.read()
    return res
    
def train_tokenizer(path: str | Path) -> Tuple[str, Tokenizer]:
    corpus = load_shakespeare(path)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab_size = 8000

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    tokenizer.train_from_iterator([corpus], trainer=trainer)

    # tokenizer.post_processor = TemplateProcessing(
    #     single="<bos> $A <eos>",
    #     pair="<bos> $A <eos> <bos> $B <eos>",
    #     special_tokens=[
    #         ("<bos>", tokenizer.token_to_id("<bos>")),
    #         ("<eos>", tokenizer.token_to_id("<eos>")),
    #     ],
    # )

    tokenizer_path = Path(path).parent / "shakespeare_bpe.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer to {str(tokenizer_path)}")

    return corpus, tokenizer

def get_batch(tokens: torch.Tensor, batch_size: int, block_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    N = tokens.size(0)
    ix = torch.randint(0, N - block_size - 1, (batch_size,), device=device)

    x = torch.stack([tokens[i:i+block_size] for i in ix])
    y = torch.stack([tokens[i+1:i+block_size+1] for i in ix])
    return x, y