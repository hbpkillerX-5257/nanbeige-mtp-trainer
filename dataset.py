import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


def get_dataloader(config, tokenizer, local_rank=0, world_size=1):
    """
    Load dataset and return DataLoader configured for multi-GPU training.
    """
    print(f"[Rank {local_rank}] Loading dataset: {config.dataset_name} ({config.dataset_config})...")
    
    try:
        if config.dataset_config:
            raw_dataset = load_dataset(config.dataset_name, config.dataset_config, split=f"train[:{config.max_samples}]")
        else:
            raw_dataset = load_dataset(config.dataset_name, split=f"train[:{config.max_samples}]")
    except Exception as e:
        print(f"Fallback to wikitext dataset due to load error: {e}")
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Extract non-empty text sequences
    texts = []
    text_key = "text" if "text" in raw_dataset.column_names else raw_dataset.column_names[0]
    for row in raw_dataset:
        val = row[text_key]
        if isinstance(val, str) and len(val.strip()) > 40:
            texts.append(val)

    def collate_fn(batch):
        encodings = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=config.max_seq_len + 2,
            return_tensors="pt"
        )
        return encodings["input_ids"]

    # In DDP / torchrun, use DistributedSampler
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            texts,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True
        )
        loader = DataLoader(
            texts,
            batch_size=config.batch_size_per_gpu,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True
        )
    else:
        loader = DataLoader(
            texts,
            batch_size=config.batch_size_per_gpu,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

    return loader
