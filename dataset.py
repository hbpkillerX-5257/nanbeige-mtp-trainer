import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def format_instruct_example(instruction, input_text, output, tokenizer):
    """
    Format one instruction row and locate the first assistant-answer token.

    `enable_thinking=False` makes the generation prefix identical to the empty
    reasoning prefix emitted for Alpaca assistant messages.
    """
    user_content = f"{instruction}\n{input_text or ''}".strip()
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": output},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception:
        prompt_text = f"User: {user_content}\n\nAssistant: "
        full_text = f"{prompt_text}{output}"

    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Chat template generation prefix does not match the full assistant "
            "conversation; cannot construct a reliable answer-only mask"
        )

    assistant_start = len(
        tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
    )
    return {
        "text": full_text,
        "assistant_start": assistant_start,
    }


def get_dataloader(config, tokenizer, rank=0, world_size=1):
    """
    Load dataset and return DataLoader configured for multi-GPU training.
    """
    print(f"[Rank {rank}] Loading dataset: {config.dataset_name} ({config.dataset_config})...")
    
    try:
        if config.dataset_config:
            raw_dataset = load_dataset(config.dataset_name, config.dataset_config, split=f"train[:{config.max_samples}]")
        else:
            raw_dataset = load_dataset(config.dataset_name, split=f"train[:{config.max_samples}]")
    except Exception as e:
        print(f"Fallback to wikitext dataset due to load error: {e}")
        raw_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Extract non-empty examples and retain the assistant-answer boundary.
    examples = []
    
    # Check if dataset has instruct columns
    is_instruct = "instruction" in raw_dataset.column_names and "output" in raw_dataset.column_names
    
    for row in raw_dataset:
        if is_instruct:
            example = format_instruct_example(
                row["instruction"],
                row.get("input", ""),
                row["output"],
                tokenizer,
            )
        else:
            text_key = "text" if "text" in raw_dataset.column_names else raw_dataset.column_names[0]
            text = row[text_key]
            example = {"text": text, "assistant_start": 0}
            
        if isinstance(example["text"], str) and len(example["text"].strip()) > 40:
            examples.append(example)

    def collate_fn(batch):
        encodings = tokenizer(
            [example["text"] for example in batch],
            padding=True,
            truncation=True,
            max_length=config.max_seq_len + 2,
            return_tensors="pt",
            add_special_tokens=False
        )
        positions = torch.arange(encodings["input_ids"].size(1)).unsqueeze(0)
        assistant_starts = torch.tensor(
            [example["assistant_start"] for example in batch],
            dtype=torch.long,
        ).unsqueeze(1)
        assistant_mask = positions >= assistant_starts
        assistant_mask &= encodings["attention_mask"].bool()
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "assistant_mask": assistant_mask,
        }

    # In DDP / torchrun, use DistributedSampler
    if world_size > 1:
        sampler = torch.utils.data.distributed.DistributedSampler(
            examples,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        loader = DataLoader(
            examples,
            batch_size=config.batch_size_per_gpu,
            sampler=sampler,
            collate_fn=collate_fn,
            drop_last=True
        )
    else:
        loader = DataLoader(
            examples,
            batch_size=config.batch_size_per_gpu,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True
        )

    return loader
