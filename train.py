import os
import sys
from pathlib import Path

# Add package directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from safetensors.torch import save_file

from config import TrainingConfig
from mtp_model import MTPModule
from dataset import get_dataloader


# Monkey-patch DynamicCache.from_legacy_cache and to_legacy_cache for newer transformers compatibility
import transformers.cache_utils
if not hasattr(transformers.cache_utils.DynamicCache, "from_legacy_cache"):
    @classmethod
    def from_legacy_cache(cls, past_key_values=None):
        if past_key_values is None:
            return cls()
        if isinstance(past_key_values, cls):
            return past_key_values
        cache = cls()
        if past_key_values is not None:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                cache.update(key_states, value_states, layer_idx)
        return cache
    transformers.cache_utils.DynamicCache.from_legacy_cache = from_legacy_cache

if not hasattr(transformers.cache_utils.DynamicCache, "to_legacy_cache"):
    def to_legacy_cache(self):
        legacy_cache = ()
        keys = getattr(self, "key_cache", getattr(self, "_key_cache", []))
        values = getattr(self, "value_cache", getattr(self, "_value_cache", []))
        for layer_idx in range(len(keys)):
            legacy_cache += ((keys[layer_idx], values[layer_idx]),)
        return legacy_cache
    transformers.cache_utils.DynamicCache.to_legacy_cache = to_legacy_cache


def init_distributed():
    """
    Initialize Distributed Data Parallel (DDP) for torchrun / multi-GPU training.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train(resume: bool = False):
    config = TrainingConfig()
    
    # Override config with command line args if specified
    if resume:
        config.resume_from_checkpoint = True
    rank, local_rank, world_size = init_distributed()
    is_main_process = (rank == 0)

    if is_main_process:
        print(f"=== Starting Distributed MTP Training (World Size: {world_size}) ===")
        os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if config.mixed_precision == "fp16" else torch.bfloat16

    # 1. Load Tokenizer & Base Model
    if is_main_process:
        print(f"Loading Base Model: {config.base_model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Config and patch rope_scaling for newer transformers compatibility
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        if isinstance(model_config.rope_scaling, dict):
            rope_type = model_config.rope_scaling.get("type", model_config.rope_scaling.get("rope_type", None))
            if rope_type is None or rope_type == "default":
                model_config.rope_scaling = None
            else:
                model_config.rope_scaling.setdefault("type", rope_type)
                model_config.rope_scaling.setdefault("factor", 1.0)

    # Load base model on current GPU rank
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        config=model_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": device}
    )

    # Freeze Base Model parameters completely
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.eval()

    hidden_size = base_model.config.hidden_size
    vocab_size = base_model.config.vocab_size

    # 2. Instantiate MTP Module
    # Initialize MTP module by explicitly cloning the base model's last layer
    mtp_module = MTPModule(
        hidden_size=hidden_size,
        base_layer=base_model.model.layers[-1]
    ).to(device=device, dtype=torch.float32)  # Force MTP head to FP32 to completely avoid NaN overflows
    
    # 2.5 Resume from Checkpoint if requested
    checkpoint_path = os.path.join(config.checkpoint_dir, "nanbeige_mtp_head.pt")
    if config.resume_from_checkpoint and os.path.exists(checkpoint_path):
        if is_main_process:
            print(f"=== Resuming training from checkpoint: {checkpoint_path} ===")
        
        # Load weights (map_location ensures they load onto the correct GPU for DDP)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        mtp_module.load_state_dict(state_dict)
        if is_main_process:
            print("=== Successfully loaded checkpoint weights! ===")


    # Wrap MTP module in PyTorch DDP for multi-GPU synchronization
    if world_size > 1:
        mtp_module = DDP(mtp_module, device_ids=[local_rank], output_device=local_rank)

    # 3. Setup Optimizer
    optimizer = torch.optim.AdamW(mtp_module.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # 4. Setup DataLoader
    dataloader = get_dataloader(config, tokenizer, local_rank=local_rank, world_size=world_size)
    
    total_steps = (len(dataloader) // config.gradient_accumulation_steps) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps)

    # 5. Training Loop
    if is_main_process:
        print(f"Training MTP Head over {config.epochs} epoch(s)...")

    raw_mtp = mtp_module.module if hasattr(mtp_module, "module") else mtp_module
    raw_mtp.train()
    optimizer.zero_grad()

    for epoch in range(config.epochs):
        if hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=not is_main_process)
        for step, input_ids in enumerate(pbar):
            input_ids = input_ids.to(device)
            if input_ids.shape[1] < 4:
                continue

            # Forward pass through base model (no gradients or KV cache needed)
            with torch.no_grad():
                outputs = base_model(input_ids, output_hidden_states=True, use_cache=False)
                # Hidden states h_t (for step t): [B, S-2, D]
                h_t = outputs.hidden_states[-1][:, :-2, :].clone().detach()
                
                # KD TARGET: Teacher's probability distribution for t+2
                # outputs.logits[i] predicts token i+1. So index 1 predicts token 2 (t+2)
                target_logits = outputs.logits[:, 1:-1, :].clone().detach()
                
                # Immediately free the base model outputs (which contains 44 layers of hidden states and logits)
                del outputs
                torch.cuda.empty_cache()
                
                # Token embeddings e(y_{t+1}) (the actual next token): [B, S-2, D]
                embed_layer = base_model.get_input_embeddings()
                emb_next = embed_layer(input_ids[:, 1:-1])
                
                # Extract ground truth targets for masking out padding
                targets = input_ids[:, 2:]

            # Forward pass through MTP module in explicit float32
            mtp_features = mtp_module(h_t.to(torch.float32), emb_next.to(torch.float32))
            
            # Predict MTP logits (cast down from fp32 to base model dtype to save 2.04 GB of VRAM!)
            lm_head = base_model.get_output_embeddings()
            mtp_logits = F.linear(mtp_features.to(lm_head.weight.dtype), lm_head.weight)            
            # KNOWLEDGE DISTILLATION LOSS (KL-Divergence)
            # PyTorch KLDiv expects log-probs for predictions, and standard probs for targets.
            mtp_log_probs = F.log_softmax(mtp_logits.to(torch.float32), dim=-1)
            teacher_probs = F.softmax(target_logits.to(torch.float32), dim=-1)
            
            # CRITICAL FIX: Mask out padding tokens so we don't compute KD on garbage!
            valid_mask = (targets != tokenizer.pad_token_id).view(-1)
            
            # Flatten to [N, Vocab] and filter out padding tokens
            mtp_log_probs_flat = mtp_log_probs.view(-1, mtp_log_probs.size(-1))[valid_mask]
            teacher_probs_flat = teacher_probs.view(-1, teacher_probs.size(-1))[valid_mask]
            
            kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
            loss = kl_loss_fn(mtp_log_probs_flat, teacher_probs_flat)
            
            # For logging: calculate how often MTP top-1 matches Teacher top-1
            teacher_preds = torch.argmax(teacher_probs_flat, dim=-1)
            mtp_preds = torch.argmax(mtp_log_probs_flat, dim=-1)
            correct_tokens = (mtp_preds == teacher_preds).sum().item()
            total_tokens = teacher_preds.numel()
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            
            # Delete massive logits tensor immediately after backward graph is populated
            del mtp_logits
            del mtp_features
            del target_logits
            del mtp_log_probs
            del teacher_probs
            del mtp_log_probs_flat
            del teacher_probs_flat
            torch.cuda.empty_cache()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(mtp_module.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if is_main_process:
                    pbar.set_postfix({
                        "loss": round(loss.item(), 4),
                        "lr": scheduler.get_last_lr()[0]
                    })

    # 6. Save Model Checkpoint (Only Main Process)
    if is_main_process:
        print("=== Saving Trained MTP Model ===")
        save_path = os.path.join(config.output_dir, config.checkpoint_name)
        
        # Save standard PyTorch weights
        torch.save(raw_mtp.state_dict(), save_path)
        print(f"Saved PyTorch weights to: {save_path}")

        # Save llama.cpp formatted state dict
        gguf_dict = raw_mtp.export_llama_cpp_state_dict()
        gguf_save_path = os.path.join(config.output_dir, "nanbeige_mtp_gguf_tensors.pt")
        torch.save(gguf_dict, gguf_save_path)
        print(f"Saved llama.cpp formatted tensors to: {gguf_save_path}")

        if config.save_safetensors:
            st_path = os.path.join(config.output_dir, "nanbeige_mtp_head.safetensors")
            save_file(gguf_dict, st_path)
            print(f"Saved Safetensors to: {st_path}")

    cleanup_distributed()
    if is_main_process:
        print("=== Training Successfully Completed ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MTP Head for Nanbeige")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint if it exists")
    args = parser.parse_args()
    
    train(resume=args.resume)
