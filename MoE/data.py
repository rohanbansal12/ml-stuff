"""
Multi-Source Dataset Loader for Qwen MoE
=========================================

A deterministic pipeline to load ~1000 samples from each of 6 dataset categories,
tokenize them consistently, and prepare them for Qwen MoE model inference.

Categories:
    - Wikipedia: General factual prose
    - Code: Programming (The Stack)
    - Conversation: Dialogue format (OpenAssistant)
    - Books: Extended narrative (PG-19)
    - Math: Mathematical reasoning (GSM8K)
    - Web text: Real-world web text (RedPajama)

Usage:
    from dataset_loader_v2 import load_multi_source_data
    
    dataloader, dataset, tokenizer = load_multi_source_data(
        samples_per_category=1000,
        max_length=2048,
        batch_size=8,
    )
    
    for batch in dataloader:
        # batch contains: input_ids, attention_mask, category_ids, source_ids
        outputs = model(**batch)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import random
import numpy as np
from functools import partial


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SourceConfig:
    """Configuration for a single data source."""
    name: str
    category: str
    hf_path: str
    hf_subset: Optional[str] = None
    split: str = "train"
    text_field: str = "text"
    text_formatter: Optional[callable] = None  # Custom text extraction function


def gsm8k_formatter(item: Dict) -> str:
    """Format GSM8K items by combining question and answer."""
    return f"Question: {item['question']}\n\nAnswer: {item['answer']}"


# Default dataset configurations
# NOTE: These have been tested to work without authentication as of 2024
DEFAULT_SOURCES: List[SourceConfig] = [
    SourceConfig(
        name="wikipedia",
        category="wikipedia",
        hf_path="wikimedia/wikipedia",
        hf_subset="20231101.simple",  # Simple English - faster to load
        text_field="text",
    ),
    SourceConfig(
        name="codeparrot",
        category="code",
        hf_path="codeparrot/codeparrot-clean",  # Public Python code dataset
        text_field="content",
    ),
    SourceConfig(
        name="openassistant",
        category="conversation",
        hf_path="OpenAssistant/oasst1",
        text_field="text",
    ),
    SourceConfig(
        name="pg19",
        category="books",
        hf_path="emozilla/pg19",  # Parquet version, no loading script needed
        text_field="text",
    ),
    SourceConfig(
        name="gsm8k",
        category="math",
        hf_path="openai/gsm8k",
        hf_subset="main",
        text_formatter=gsm8k_formatter,
    ),
    SourceConfig(
        name="c4",
        category="web_text",
        hf_path="allenai/c4",
        hf_subset="en",  # English subset
        text_field="text",
    ),
]


# =============================================================================
# Utility Functions
# =============================================================================

def set_deterministic(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For completely deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_text(item: Dict, config: SourceConfig) -> Optional[str]:
    """Extract text from a dataset item based on config."""
    if config.text_formatter:
        text = config.text_formatter(item)
    else:
        text = item.get(config.text_field, "")
    
    # Return None for empty/invalid text
    if not text or not isinstance(text, str) or not text.strip():
        return None
    return text.strip()


# =============================================================================
# Data Loading
# =============================================================================

def load_source_samples(
    config: SourceConfig,
    n_samples: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load and sample from a single data source.
    
    Returns list of dicts: {'text': str, 'source': str, 'category': str}
    """
    print(f"  Loading {config.name}...", end=" ", flush=True)
    
    try:
        # Load dataset - use streaming for very large datasets
        load_kwargs = {
            "path": config.hf_path,
            "split": config.split,
        }
        if config.hf_subset:
            load_kwargs["name"] = config.hf_subset
        
        # Use streaming for large datasets to avoid downloading everything
        large_datasets = {"allenai/c4", "codeparrot/codeparrot-clean"}
        use_streaming = config.hf_path in large_datasets
        
        if use_streaming:
            load_kwargs["streaming"] = True
            ds = load_dataset(**load_kwargs)
            
            # For streaming, we need to manually sample
            # Set seed for reproducibility
            rng = random.Random(seed)
            
            samples = []
            # Take more than needed to account for filtering
            buffer_size = n_samples * 3
            buffer = []
            
            for i, item in enumerate(ds):
                if i >= buffer_size:
                    break
                text = extract_text(item, config)
                if text:
                    buffer.append({
                        "text": text,
                        "source": config.name,
                        "category": config.category,
                    })
            
            # Shuffle and take n_samples
            rng.shuffle(buffer)
            samples = buffer[:n_samples]
        else:
            ds = load_dataset(**load_kwargs)
            
            # Deterministic shuffle
            ds = ds.shuffle(seed=seed)
            
            # Sample (handle datasets smaller than n_samples)
            actual_n = min(n_samples, len(ds))
            ds = ds.select(range(actual_n))
            
            # Extract text and create samples
            samples = []
            for item in ds:
                text = extract_text(item, config)
                if text:
                    samples.append({
                        "text": text,
                        "source": config.name,
                        "category": config.category,
                    })
        
        print(f"✓ ({len(samples)} samples)")
        return samples
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return []


def load_all_sources(
    sources: List[SourceConfig],
    samples_per_source: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load samples from all configured sources."""
    print("Loading datasets:")
    
    all_samples = []
    for config in sources:
        samples = load_source_samples(config, samples_per_source, seed)
        all_samples.extend(samples)
    
    print(f"\nTotal: {len(all_samples)} samples loaded")
    return all_samples


# =============================================================================
# Dataset Class
# =============================================================================

class MultiSourceTokenizedDataset(Dataset):
    """
    PyTorch Dataset with pre-tokenized samples and source tracking.
    
    Attributes:
        categories: List of unique category names
        sources: List of unique source names
        category_to_id: Mapping from category name to integer ID
        source_to_id: Mapping from source name to integer ID
    """
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Build category/source mappings
        self.categories = sorted(set(s["category"] for s in samples))
        self.sources = sorted(set(s["source"] for s in samples))
        self.category_to_id = {c: i for i, c in enumerate(self.categories)}
        self.source_to_id = {s: i for i, s in enumerate(self.sources)}
        self.id_to_category = {i: c for c, i in self.category_to_id.items()}
        self.id_to_source = {i: s for s, i in self.source_to_id.items()}
        
        # Tokenize all samples
        self.data = self._tokenize_samples(samples)
    
    def _tokenize_samples(self, samples: List[Dict]) -> List[Dict]:
        """Tokenize all samples with progress reporting."""
        print(f"Tokenizing {len(samples)} samples...")
        
        tokenized = []
        for i, sample in enumerate(samples):
            enc = self.tokenizer(
                sample["text"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors='pt',
            )
            
            tokenized.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "category_id": self.category_to_id[sample["category"]],
                "source_id": self.source_to_id[sample["source"]],
                "idx": i,
            })
            
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(samples)} tokenized...")
        
        print("Tokenization complete.")
        return tokenized
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]
    
    def get_indices_by_category(self, category: str) -> List[int]:
        """Get all dataset indices for a specific category."""
        cat_id = self.category_to_id[category]
        return [i for i, d in enumerate(self.data) if d["category_id"] == cat_id]
    
    def get_indices_by_source(self, source: str) -> List[int]:
        """Get all dataset indices for a specific source."""
        src_id = self.source_to_id[source]
        return [i for i, d in enumerate(self.data) if d["source_id"] == src_id]
    
    def summary(self) -> Dict:
        """Get dataset summary statistics."""
        stats = {
            "total": len(self),
            "by_category": {c: 0 for c in self.categories},
            "by_source": {s: 0 for s in self.sources},
            "avg_tokens": 0,
            "max_tokens": 0,
            "min_tokens": float("inf"),
        }
        
        total_tokens = 0
        for d in self.data:
            n_tokens = len(d["input_ids"])
            total_tokens += n_tokens
            stats["max_tokens"] = max(stats["max_tokens"], n_tokens)
            stats["min_tokens"] = min(stats["min_tokens"], n_tokens)
            
            cat = self.id_to_category[d["category_id"]]
            src = self.id_to_source[d["source_id"]]
            stats["by_category"][cat] += 1
            stats["by_source"][src] += 1
        
        stats["avg_tokens"] = total_tokens / len(self) if self.data else 0
        return stats


# =============================================================================
# Collate Function & DataLoader
# =============================================================================

def collate_with_padding(
    batch: List[Dict],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads sequences to batch max length.
    
    Returns dict with:
        - input_ids: (batch_size, seq_len)
        - attention_mask: (batch_size, seq_len)
        - category_ids: (batch_size,)
        - source_ids: (batch_size,)
        - indices: (batch_size,) - original dataset indices
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    
    input_ids = []
    attention_mask = []
    category_ids = []
    source_ids = []
    indices = []
    
    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len
        
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        category_ids.append(item["category_id"])
        source_ids.append(item["source_id"])
        indices.append(item["idx"])
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "category_ids": torch.tensor(category_ids, dtype=torch.long),
        "source_ids": torch.tensor(source_ids, dtype=torch.long),
        "indices": torch.tensor(indices, dtype=torch.long),
    }


# =============================================================================
# Main API
# =============================================================================

def load_multi_source_data(
    tokenizer_name: str = "Qwen/Qwen1.5-MoE-A2.7B",
    samples_per_category: int = 1000,
    max_length: int = 2048,
    batch_size: int = 8,
    seed: int = 42,
    shuffle_batches: bool = False,
    sources: Optional[List[SourceConfig]] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, MultiSourceTokenizedDataset, AutoTokenizer]:
    """
    Main function to create a DataLoader with tokenized multi-source data.
    
    Args:
        tokenizer_name: HuggingFace model/tokenizer name (default: Qwen MoE)
        samples_per_category: Target samples per data source (~1000)
        max_length: Maximum token length for truncation
        batch_size: DataLoader batch size
        seed: Random seed for reproducibility
        shuffle_batches: Whether to shuffle batches (False keeps deterministic order)
        sources: Custom source configs (uses defaults if None)
        num_workers: DataLoader workers (0 for main process)
    
    Returns:
        Tuple of (DataLoader, Dataset, Tokenizer)
    
    Example:
        dataloader, dataset, tokenizer = load_multi_source_data()
        
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            # Track which samples came from which category
            categories = batch["category_ids"]
    """
    # Set deterministic behavior
    set_deterministic(seed)
    
    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load all samples
    source_configs = sources or DEFAULT_SOURCES
    samples = load_all_sources(source_configs, samples_per_category, seed)
    
    # Create dataset
    dataset = MultiSourceTokenizedDataset(samples, tokenizer, max_length)
    
    # Create dataloader
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_batches,
        num_workers=num_workers,
        collate_fn=partial(collate_with_padding, pad_token_id=pad_id),
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed) if shuffle_batches else None,
    )
    
    return dataloader, dataset, tokenizer


# =============================================================================
# Usage Examples
# =============================================================================

def demo_basic_usage():
    """Demonstrate basic usage of the data loader."""
    print("=" * 70)
    print("Basic Usage Demo")
    print("=" * 70)
    
    # Load data (smaller sample for demo)
    dataloader, dataset, tokenizer = load_multi_source_data(
        samples_per_category=50,  # Small for demo
        max_length=512,
        batch_size=4,
        seed=42,
    )
    
    # Print summary
    print("\n" + "-" * 40)
    print("Dataset Summary:")
    print("-" * 40)
    summary = dataset.summary()
    print(f"Total samples: {summary['total']}")
    print(f"Token stats: min={summary['min_tokens']}, avg={summary['avg_tokens']:.0f}, max={summary['max_tokens']}")
    print("\nBy category:")
    for cat, count in summary["by_category"].items():
        print(f"  {cat}: {count}")
    
    # Iterate through a few batches
    print("\n" + "-" * 40)
    print("Sample Batches:")
    print("-" * 40)
    
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
        
        print(f"\nBatch {i}:")
        print(f"  Shape: {batch['input_ids'].shape}")
        
        # Decode category/source IDs back to names
        cats = [dataset.id_to_category[c.item()] for c in batch["category_ids"]]
        srcs = [dataset.id_to_source[s.item()] for s in batch["source_ids"]]
        print(f"  Categories: {cats}")
        print(f"  Sources: {srcs}")


def demo_model_inference():
    """Show how to use with a Qwen MoE model."""
    print("\n" + "=" * 70)
    print("Model Inference Pattern (pseudocode)")
    print("=" * 70)
    
    code = '''
from transformers import AutoModelForCausalLM
from dataset_loader_v2 import load_multi_source_data

# Load model and data
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    trust_remote_code=True,
    device_map="auto",
)

dataloader, dataset, tokenizer = load_multi_source_data(
    samples_per_category=1000,
    max_length=2048,
    batch_size=8,
)

# Inference loop with category tracking
results_by_category = {cat: [] for cat in dataset.categories}

for batch in dataloader:
    # Move to device
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    
    # Store results by category
    for i, cat_id in enumerate(batch["category_ids"]):
        category = dataset.id_to_category[cat_id.item()]
        results_by_category[category].append({
            "logits": outputs.logits[i],
            "hidden_state": outputs.hidden_states[-1][i],
        })

# Analyze per-category
for category, results in results_by_category.items():
    print(f"{category}: {len(results)} samples processed")
'''
    print(code)


def demo_category_specific_analysis():
    """Show how to analyze specific categories."""
    print("\n" + "=" * 70)
    print("Category-Specific Analysis")
    print("=" * 70)
    
    # Load data
    dataloader, dataset, tokenizer = load_multi_source_data(
        samples_per_category=100,
        max_length=256,
        batch_size=8,
        seed=42,
    )
    
    # Get indices for specific categories
    print("\nAccessing specific categories:")
    for category in dataset.categories:
        indices = dataset.get_indices_by_category(category)
        print(f"  {category}: {len(indices)} samples (indices: {indices[:3]}...)")
    
    # Example: iterate only over code samples
    print("\nIterating over 'code' samples only:")
    code_indices = dataset.get_indices_by_category("code")
    code_subset = torch.utils.data.Subset(dataset, code_indices)
    code_loader = DataLoader(
        code_subset,
        batch_size=4,
        collate_fn=partial(
            collate_with_padding,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        ),
    )
    
    for i, batch in enumerate(code_loader):
        if i >= 1:
            break
        print(f"  Code batch shape: {batch['input_ids'].shape}")
        # All samples in this batch are code
        print(f"  All category_ids are 'code': {all(c == dataset.category_to_id['code'] for c in batch['category_ids'].tolist())}")


if __name__ == "__main__":
    demo_basic_usage()
    demo_model_inference()
    demo_category_specific_analysis()
    print("\n✓ All demos complete!")