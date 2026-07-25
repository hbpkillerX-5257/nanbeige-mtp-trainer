import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add package directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Nanbeige's remote module imports this symbol from transformers.utils. Patch
# both namespaces before loading the remote module so a broken flash-attn
# installation cannot be selected accidentally.
import transformers.utils
import transformers.utils.import_utils

transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
transformers.utils.is_flash_attn_2_available = lambda: False

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import save_file
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from config import TrainingConfig
from dataset import get_dataloader
from mtp_model import MTPModule


REQUIRED_TRANSFORMERS_VERSION = "4.42.4"


def require_compatible_transformers() -> None:
    if transformers.__version__ != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"Nanbeige4.2 requires transformers=={REQUIRED_TRANSFORMERS_VERSION}, "
            f"but {transformers.__version__} is installed. Install the pinned "
            "version with: pip install --upgrade "
            f"'transformers=={REQUIRED_TRANSFORMERS_VERSION}'"
        )


def init_distributed() -> Tuple[int, int, int]:
    """Initialize DDP for torchrun, or return a single-process configuration."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 0, 1

    if not torch.cuda.is_available():
        raise RuntimeError("torchrun/DDP training requires CUDA for the NCCL backend")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_ranks_true(local_value: bool, device: torch.device) -> bool:
    """Return True only when every DDP rank reports True."""
    if not dist.is_initialized():
        return local_value

    flag = torch.tensor(int(local_value), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def resolve_model_dtype(config: TrainingConfig, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32

    precision = config.mixed_precision.lower()
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Nanbeige4.2 requires BF16, but this GPU does not support it. "
                "NVIDIA T4/P100 GPUs are not suitable; use an A100, L4, A10, "
                "RTX 30/40-series, or newer BF16-capable GPU."
            )
        return torch.bfloat16
    if precision == "fp16":
        raise RuntimeError(
            "FP16 is disabled for Nanbeige4.2 because it collapses the teacher "
            "predictions. Use mixed_precision='bf16' on supported hardware, or "
            "'fp32' if sufficient memory is available."
        )
    if precision == "fp32":
        return torch.float32
    raise ValueError("mixed_precision must be one of: fp16, bf16, fp32")


def normalize_nanbeige_rope_scaling(model_config) -> None:
    """
    Translate current Transformers RoPE metadata to Nanbeige's legacy format.

    Recent Transformers versions may turn the model's default RoPE settings
    into {"rope_type": "default", ...}. Nanbeige treats any non-None value as
    scaled RoPE and expects the legacy {"type", "factor"} keys.
    """
    rope_scaling = getattr(model_config, "rope_scaling", None)
    if not isinstance(rope_scaling, dict):
        return

    scaling_type = rope_scaling.get("type", rope_scaling.get("rope_type"))
    if scaling_type in (None, "default"):
        model_config.rope_scaling = None
        return

    if scaling_type not in {"linear", "dynamic"}:
        raise ValueError(
            f"Nanbeige remote code does not support rope scaling type {scaling_type!r}"
        )
    if "factor" not in rope_scaling:
        raise ValueError(f"RoPE scaling type {scaling_type!r} requires a factor")

    normalized = dict(rope_scaling)
    normalized["type"] = scaling_type
    model_config.rope_scaling = normalized


@torch.no_grad()
def chunked_top1(
    features: torch.Tensor,
    lm_head_weight: torch.Tensor,
    vocab_chunk_size: int,
) -> torch.Tensor:
    """Return exact LM-head top-1 IDs without allocating full-vocabulary logits."""
    if vocab_chunk_size <= 0:
        raise ValueError("kd_vocab_chunk_size must be positive")

    original_shape = features.shape[:-1]
    flat_features = features.reshape(-1, features.size(-1)).to(lm_head_weight.dtype)
    best_values = torch.full(
        (flat_features.size(0),),
        -torch.inf,
        device=features.device,
        dtype=torch.float32,
    )
    best_ids = torch.zeros(
        flat_features.size(0),
        device=features.device,
        dtype=torch.long,
    )

    for start in range(0, lm_head_weight.size(0), vocab_chunk_size):
        end = min(start + vocab_chunk_size, lm_head_weight.size(0))
        logits = F.linear(flat_features, lm_head_weight[start:end]).float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Base model produced non-finite LM-head logits")
        chunk_values, chunk_ids = logits.max(dim=-1)
        replace = chunk_values > best_values
        best_values = torch.maximum(best_values, chunk_values)
        best_ids[replace] = chunk_ids[replace] + start

    return best_ids.reshape(original_shape)


@torch.no_grad()
def validate_teacher_health(
    base_model,
    tokenizer,
    device: torch.device,
    vocab_chunk_size: int,
) -> Tuple[float, int, int]:
    """Fail before training if the frozen teacher has collapsed predictions."""
    messages = [
        {
            "role": "user",
            "content": "Briefly explain what artificial intelligence is.",
        },
        {
            "role": "assistant",
            "content": (
                "Artificial intelligence is the study of computer systems that "
                "perform tasks involving reasoning, learning, and decision making."
            ),
        },
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception:
        text = (
            f"User: {messages[0]['content']}\n\n"
            f"Assistant: {messages[1]['content']}"
        )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    outputs = base_model.model(
        input_ids=inputs["input_ids"],
        # The official Nanbeige quickstart supplies only input_ids for an
        # unpadded sequence.
        attention_mask=None,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    if not torch.isfinite(hidden).all():
        raise RuntimeError("Base teacher produced non-finite hidden states")

    predictions = chunked_top1(
        hidden[:, :-1, :],
        base_model.get_output_embeddings().weight,
        vocab_chunk_size,
    )
    targets = inputs["input_ids"][:, 1:]
    total = int(targets.numel())
    matches = int((predictions == targets).sum().item())
    accuracy = matches / max(total, 1)
    unique_predictions = int(predictions.unique().numel())
    minimum_unique = max(8, total // 8)

    if unique_predictions < minimum_unique:
        raise RuntimeError(
            "Base teacher health check failed: "
            f"next-token accuracy={accuracy:.2%}, "
            f"unique_predictions={unique_predictions}/{total}. "
            "The teacher has collapsed, so MTP training would be invalid. "
            "Check BF16 support and use the Transformers/model-code versions "
            "recommended by Nanbeige."
        )
    return accuracy, unique_predictions, total


@torch.no_grad()
def exact_chunked_kd(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    lm_head_weight: torch.Tensor,
    valid_mask: torch.Tensor,
    vocab_chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Compute exact mean KL(teacher || student) without materializing [N, vocab].

    The LM head is frozen, so the derivative with respect to student features is
    W^T (p_student - p_teacher). Computing that derivative explicitly lets us
    stream over vocabulary chunks and then backpropagate once through the
    trainable MTP block. Projections use the frozen head's model dtype (matching
    normal inference); normalization, KL accumulation, and gradient accumulation
    use FP32.
    """
    if vocab_chunk_size <= 0:
        raise ValueError("kd_vocab_chunk_size must be positive")

    flat_mask = valid_mask.reshape(-1)
    num_valid = int(flat_mask.sum().item())
    if num_valid == 0:
        empty_grad = torch.zeros_like(student_features, dtype=torch.float32)
        return student_features.new_zeros((), dtype=torch.float32), empty_grad, 0, 0

    projection_dtype = lm_head_weight.dtype
    student_valid = (
        student_features.detach()
        .reshape(-1, student_features.size(-1))[flat_mask]
        .to(projection_dtype)
    )
    teacher_valid = (
        teacher_features.detach()
        .reshape(-1, teacher_features.size(-1))[flat_mask]
        .to(projection_dtype)
    )
    vocab_size = lm_head_weight.size(0)
    device = student_features.device

    student_log_z = torch.full((num_valid,), -torch.inf, device=device, dtype=torch.float32)
    teacher_log_z = torch.full_like(student_log_z, -torch.inf)
    student_max = torch.full_like(student_log_z, -torch.inf)
    teacher_max = torch.full_like(student_log_z, -torch.inf)
    student_argmax = torch.zeros(num_valid, device=device, dtype=torch.long)
    teacher_argmax = torch.zeros_like(student_argmax)

    # First pass: exact normalizers and top-1 predictions.
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        weight = lm_head_weight[start:end].detach()
        student_logits = F.linear(student_valid, weight).float()
        teacher_logits = F.linear(teacher_valid, weight).float()

        student_log_z = torch.logaddexp(student_log_z, torch.logsumexp(student_logits, dim=-1))
        teacher_log_z = torch.logaddexp(teacher_log_z, torch.logsumexp(teacher_logits, dim=-1))

        chunk_student_max, chunk_student_idx = student_logits.max(dim=-1)
        chunk_teacher_max, chunk_teacher_idx = teacher_logits.max(dim=-1)
        replace_student = chunk_student_max > student_max
        replace_teacher = chunk_teacher_max > teacher_max
        student_max = torch.maximum(student_max, chunk_student_max)
        teacher_max = torch.maximum(teacher_max, chunk_teacher_max)
        student_argmax[replace_student] = chunk_student_idx[replace_student] + start
        teacher_argmax[replace_teacher] = chunk_teacher_idx[replace_teacher] + start

        del weight, student_logits, teacher_logits

    # Second pass: KL value and its exact gradient with respect to MTP features.
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    grad_valid = torch.zeros(
        student_valid.shape,
        device=device,
        dtype=torch.float32,
    )
    # Scaling before the frozen FP16/BF16 projection protects small probability
    # differences from underflow; the result is unscaled in FP32.
    projection_scale = 1024.0 if projection_dtype != torch.float32 else 1.0
    for start in range(0, vocab_size, vocab_chunk_size):
        end = min(start + vocab_chunk_size, vocab_size)
        weight = lm_head_weight[start:end].detach()
        student_logits = F.linear(student_valid, weight).float()
        teacher_logits = F.linear(teacher_valid, weight).float()
        student_log_probs = student_logits - student_log_z[:, None]
        teacher_log_probs = teacher_logits - teacher_log_z[:, None]
        teacher_probs = teacher_log_probs.exp()
        student_probs = student_log_probs.exp()

        loss_sum += (teacher_probs * (teacher_log_probs - student_log_probs)).sum()
        probability_delta = (student_probs - teacher_probs) * projection_scale
        grad_chunk = F.linear(
            probability_delta.to(projection_dtype),
            weight.t(),
        ).float()
        grad_valid.add_(grad_chunk / projection_scale)

        del (
            weight,
            student_logits,
            teacher_logits,
            student_log_probs,
            teacher_log_probs,
            teacher_probs,
            student_probs,
            probability_delta,
            grad_chunk,
        )

    loss = loss_sum / num_valid
    grad_valid.div_(num_valid)
    grad_features = torch.zeros_like(student_features, dtype=torch.float32)
    grad_features.reshape(-1, grad_features.size(-1))[flat_mask] = grad_valid
    correct = int((student_argmax == teacher_argmax).sum().item())
    return loss, grad_features, correct, num_valid


def optimizer_step(
    mtp_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    accumulated_batches: int,
) -> None:
    """Average accumulated batch gradients, clip, and update once."""
    if accumulated_batches <= 0:
        return
    for parameter in mtp_module.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(accumulated_batches)
    torch.nn.utils.clip_grad_norm_(mtp_module.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def save_trainer_state(
    path: str,
    raw_mtp: MTPModule,
    optimizer: torch.optim.Optimizer,
    scheduler,
    next_epoch: int,
    global_step: int,
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": raw_mtp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "next_epoch": next_epoch,
            "global_step": global_step,
        },
        path,
    )


def train(resume: bool = False) -> None:
    require_compatible_transformers()
    config = TrainingConfig()
    config.resume_from_checkpoint = resume
    rank, local_rank, world_size = init_distributed()
    is_main_process = rank == 0

    try:
        if is_main_process:
            print(f"=== Starting Distributed MTP Training (World Size: {world_size}) ===")
            os.makedirs(config.output_dir, exist_ok=True)
            os.makedirs(config.checkpoint_dir, exist_ok=True)

        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        dtype = resolve_model_dtype(config, device)

        if is_main_process:
            print(f"Loading Base Model: {config.base_model_name} ({dtype})...")

        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model_name,
            revision=config.model_revision,
            trust_remote_code=True,
            use_fast=False,
        )
        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
            elif tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                raise ValueError("Tokenizer has no existing token that can be used for padding")
        tokenizer.padding_side = "right"

        model_config = AutoConfig.from_pretrained(
            config.base_model_name,
            revision=config.model_revision,
            trust_remote_code=True,
        )
        normalize_nanbeige_rope_scaling(model_config)
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            config=model_config,
            revision=config.model_revision,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map={"": device},
        )
        for parameter in base_model.parameters():
            parameter.requires_grad = False
        base_model.eval()
        if is_main_process:
            print(
                f"Transformers {transformers.__version__}; "
                "attention implementation: "
                f"{getattr(base_model.config, '_attn_implementation', 'unknown')}"
            )

        teacher_accuracy, teacher_unique, teacher_tokens = validate_teacher_health(
            base_model,
            tokenizer,
            device,
            config.kd_vocab_chunk_size,
        )
        if is_main_process:
            print(
                "Teacher health check passed: "
                f"next-token accuracy={teacher_accuracy:.2%}, "
                f"unique_predictions={teacher_unique}/{teacher_tokens}"
            )

        mtp_module = MTPModule(
            hidden_size=base_model.config.hidden_size,
            base_layer=base_model.model.layers[-1],
        ).to(device=device, dtype=torch.float32)

        trainer_state_path = os.path.join(config.checkpoint_dir, config.trainer_state_name)
        legacy_weights_path = os.path.join(config.checkpoint_dir, config.checkpoint_name)
        resume_state: Optional[Dict] = None
        if config.resume_from_checkpoint:
            if os.path.exists(trainer_state_path):
                resume_state = torch.load(
                    trainer_state_path,
                    map_location=device,
                    weights_only=True,
                )
                mtp_module.load_state_dict(resume_state["model"])
                if is_main_process:
                    print(f"Loaded trainer checkpoint: {trainer_state_path}")
            elif os.path.exists(legacy_weights_path):
                mtp_module.load_state_dict(
                    torch.load(legacy_weights_path, map_location=device, weights_only=True)
                )
                if is_main_process:
                    print(
                        f"Loaded legacy MTP weights from {legacy_weights_path}; "
                        "optimizer and scheduler start fresh."
                    )
            else:
                raise FileNotFoundError(
                    f"--resume requested, but neither {trainer_state_path} nor "
                    f"{legacy_weights_path} exists"
                )

        if world_size > 1:
            mtp_module = DDP(mtp_module, device_ids=[local_rank], output_device=local_rank)
        raw_mtp = mtp_module.module if isinstance(mtp_module, DDP) else mtp_module
        mtp_module.train()

        optimizer = torch.optim.AdamW(
            mtp_module.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        dataloader = get_dataloader(config, tokenizer, rank=rank, world_size=world_size)
        updates_per_epoch = math.ceil(len(dataloader) / config.gradient_accumulation_steps)
        total_steps = max(1, updates_per_epoch * config.epochs)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        start_epoch = 0
        global_step = 0
        if resume_state is not None:
            optimizer.load_state_dict(resume_state["optimizer"])
            scheduler.load_state_dict(resume_state["scheduler"])
            start_epoch = int(resume_state.get("next_epoch", 0))
            global_step = int(resume_state.get("global_step", 0))

        if is_main_process:
            print(
                f"Training MTP head from epoch {start_epoch + 1} "
                f"through {config.epochs}..."
            )

        optimizer.zero_grad(set_to_none=True)
        lm_head = base_model.get_output_embeddings()

        for epoch in range(start_epoch, config.epochs):
            if hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)

            accumulated_batches = 0
            last_loss = None
            pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", disable=not is_main_process)
            for step, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                assistant_mask = batch["assistant_mask"].to(device, non_blocking=True)

                locally_usable = input_ids.shape[1] >= 4 and bool(
                    assistant_mask[:, 2:].any().item()
                )
                if not all_ranks_true(locally_usable, device):
                    continue

                with torch.no_grad():
                    base_outputs = base_model.model(
                        input_ids=input_ids,
                        # Follow Nanbeige's official unmasked path when this
                        # batch has no padding. A mask is needed only for
                        # genuinely padded batches.
                        attention_mask=(
                            None if bool(attention_mask.all().item()) else attention_mask
                        ),
                        output_hidden_states=False,
                        use_cache=False,
                        return_dict=True,
                    )
                    final_hidden = base_outputs.last_hidden_state
                    h_t = final_hidden[:, :-2, :]
                    teacher_features = final_hidden[:, 1:-1, :]
                    emb_next = base_model.get_input_embeddings()(input_ids[:, 1:-1])
                    mtp_attention_mask = attention_mask[:, :-2]
                    # For instruction datasets, distill only teacher
                    # probabilities whose target token is in the assistant
                    # answer. Plain-text datasets mark every token as valid.
                    valid_mask = attention_mask[:, 2:].bool()
                    valid_mask &= assistant_mask[:, 2:]

                mtp_features = mtp_module(
                    h_t.float(),
                    emb_next.float(),
                    attention_mask=mtp_attention_mask,
                )
                loss, grad_features, correct_tokens, total_tokens = exact_chunked_kd(
                    student_features=mtp_features,
                    teacher_features=teacher_features,
                    lm_head_weight=lm_head.weight,
                    valid_mask=valid_mask,
                    vocab_chunk_size=config.kd_vocab_chunk_size,
                )

                locally_finite = bool(
                    torch.isfinite(loss).item() and torch.isfinite(grad_features).all().item()
                )
                if not all_ranks_true(locally_finite, device):
                    if is_main_process:
                        print(f"Skipping non-finite batch at epoch {epoch + 1}, step {step}")
                    # Complete DDP's reducer lifecycle on every rank, then
                    # discard this entire accumulation group.
                    mtp_features.backward(torch.zeros_like(mtp_features))
                    optimizer.zero_grad(set_to_none=True)
                    accumulated_batches = 0
                    continue

                mtp_features.backward(grad_features)
                accumulated_batches += 1
                last_loss = float(loss.item())

                if is_main_process and step % 50 == 0:
                    accuracy = correct_tokens / max(total_tokens, 1)
                    pbar.write(
                        f"step={step} loss={last_loss:.4f} "
                        f"teacher_match={accuracy:.2%}"
                    )

                if accumulated_batches == config.gradient_accumulation_steps:
                    optimizer_step(
                        mtp_module,
                        optimizer,
                        scheduler,
                        accumulated_batches,
                    )
                    accumulated_batches = 0
                    global_step += 1
                    if is_main_process:
                        pbar.set_postfix(
                            loss=round(last_loss, 4),
                            lr=scheduler.get_last_lr()[0],
                        )

                del (
                    base_outputs,
                    final_hidden,
                    h_t,
                    teacher_features,
                    emb_next,
                    mtp_features,
                    grad_features,
                )

            # Flush a partial accumulation group instead of dropping it or
            # carrying it into the next epoch.
            if accumulated_batches:
                optimizer_step(
                    mtp_module,
                    optimizer,
                    scheduler,
                    accumulated_batches,
                )
                global_step += 1

            if is_main_process:
                save_trainer_state(
                    trainer_state_path,
                    raw_mtp,
                    optimizer,
                    scheduler,
                    next_epoch=epoch + 1,
                    global_step=global_step,
                )
            if dist.is_initialized():
                dist.barrier()

        if is_main_process:
            print("=== Saving Trained MTP Model ===")
            save_path = os.path.join(config.output_dir, config.checkpoint_name)
            torch.save(raw_mtp.state_dict(), save_path)
            print(f"Saved PyTorch weights to: {save_path}")

            gguf_dict = raw_mtp.export_llama_cpp_state_dict()
            gguf_save_path = os.path.join(config.output_dir, "nanbeige_mtp_gguf_tensors.pt")
            torch.save(gguf_dict, gguf_save_path)
            print(f"Saved llama.cpp formatted tensors to: {gguf_save_path}")

            if config.save_safetensors:
                st_path = os.path.join(config.output_dir, "nanbeige_mtp_head.safetensors")
                save_file(gguf_dict, st_path)
                print(f"Saved Safetensors to: {st_path}")

            print("=== Training Successfully Completed ===")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MTP Head for Nanbeige")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model, optimizer, scheduler, epoch, and step from trainer_state.pt",
    )
    args = parser.parse_args()
    train(resume=args.resume)
