import argparse
from typing import Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from config import TrainingConfig
from dataset import format_instruct_example
from mtp_model import MTPModule
from train import (
    chunked_top1,
    normalize_nanbeige_rope_scaling,
    resolve_model_dtype,
)


def load_base_model(
    model_name: str,
    device: torch.device,
    training_config: TrainingConfig,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load tokenizer and teacher with the same compatibility settings as training."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
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

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    normalize_nanbeige_rope_scaling(model_config)
    dtype = resolve_model_dtype(training_config, device)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=model_config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map={"": device},
    )
    model.eval()
    return tokenizer, model


@torch.no_grad()
def evaluate_acceptance_rate(
    base_model_name: str = "Nanbeige/Nanbeige4.2-3B",
    mtp_weights_path: str = "mtp_output/nanbeige_mtp_head.pt",
    text_sample: str = (
        "The history of artificial intelligence began in antiquity, with myths "
        "and stories of artificial beings endowed with intelligence. The modern "
        "field emerged from advances in logic, computation, neuroscience, and "
        "the formal study of human reasoning."
    ),
    device_name: str = "cuda:0",
) -> None:
    training_config = TrainingConfig()
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    print("=== Loading Tokenizer and Base Model ===")
    tokenizer, base_model = load_base_model(
        base_model_name,
        device,
        training_config,
    )

    print(f"=== Loading MTP Head from {mtp_weights_path} ===")
    mtp_module = MTPModule(
        hidden_size=base_model.config.hidden_size,
        base_layer=base_model.model.layers[-1],
    ).to(device=device, dtype=torch.float32)
    state_dict = torch.load(mtp_weights_path, map_location=device, weights_only=True)
    mtp_module.load_state_dict(state_dict)
    mtp_module.eval()

    example = format_instruct_example(
        "Write a short overview of the history of artificial intelligence.",
        "",
        text_sample,
        tokenizer,
    )
    inputs = tokenizer(
        example["text"],
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    positions = torch.arange(input_ids.size(1), device=device).unsqueeze(0)
    assistant_mask = positions >= example["assistant_start"]
    assistant_mask &= attention_mask.bool()
    if input_ids.size(1) < 4:
        raise ValueError("Evaluation sample must contain at least four tokens")

    print("=== Running Evaluation ===")
    base_outputs = base_model.model(
        input_ids=input_ids,
        attention_mask=None if bool(attention_mask.all().item()) else attention_mask,
        output_hidden_states=False,
        use_cache=False,
        return_dict=True,
    )
    final_hidden = base_outputs.last_hidden_state

    # This exactly matches training:
    # h_t + embedding(y_{t+1}) -> MTP prediction for the teacher at t+2.
    h_t = final_hidden[:, :-2, :]
    teacher_features = final_hidden[:, 1:-1, :]
    emb_next = base_model.get_input_embeddings()(input_ids[:, 1:-1])
    mtp_attention_mask = attention_mask[:, :-2]
    valid_mask = attention_mask[:, 2:].bool()
    valid_mask &= assistant_mask[:, 2:]

    mtp_features = mtp_module(
        h_t.float(),
        emb_next.float(),
        attention_mask=mtp_attention_mask,
    )
    lm_head_weight = base_model.get_output_embeddings().weight
    teacher_preds = chunked_top1(
        teacher_features,
        lm_head_weight,
        training_config.kd_vocab_chunk_size,
    )
    mtp_preds = chunked_top1(
        mtp_features,
        lm_head_weight,
        training_config.kd_vocab_chunk_size,
    )

    valid_teacher = teacher_preds[valid_mask]
    valid_mtp = mtp_preds[valid_mask]
    valid_targets = input_ids[:, 2:][valid_mask]
    correct = int((valid_mtp == valid_teacher).sum().item())
    total = int(valid_teacher.numel())
    acceptance_rate = 100.0 * correct / max(total, 1)
    teacher_target_rate = 100.0 * int(
        (valid_teacher == valid_targets).sum().item()
    ) / max(total, 1)
    unique_teacher = int(valid_teacher.unique().numel())
    unique_mtp = int(valid_mtp.unique().numel())

    print("\n" + "=" * 58)
    print(f"Total tokens evaluated:              {total}")
    print(f"MTP tokens matching teacher:         {correct}")
    print(f"Teacher/MTP top-1 agreement:         {acceptance_rate:.2f}%")
    print(f"Teacher top-1 vs actual next token:  {teacher_target_rate:.2f}%")
    print(f"Unique teacher top-1 predictions:    {unique_teacher}")
    print(f"Unique MTP top-1 predictions:        {unique_mtp}")
    print("=" * 58)

    if unique_teacher <= 2 and total >= 20:
        print(
            "\nWARNING: The base teacher has collapsed to almost constant "
            "predictions. Do not use the MTP agreement score until the base "
            "model precision/attention path is healthy."
        )

    print("\nSample Predictions:")
    valid_positions = valid_mask[0].nonzero(as_tuple=False).flatten()
    for position in valid_positions[:10].tolist():
        context_end = position + 2
        teacher_id = int(teacher_preds[0, position].item())
        mtp_id = int(mtp_preds[0, position].item())
        context = tokenizer.decode(input_ids[0, :context_end])
        teacher_token = tokenizer.decode([teacher_id])
        mtp_token = tokenizer.decode([mtp_id])
        match = "✅" if teacher_id == mtp_id else "❌"
        print(f"Context: {context!r}")
        print(f"  Base Model Predicts: {teacher_token!r}")
        print(f"  MTP Head Predicts  : {mtp_token!r} {match}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate teacher-forced two-token MTP agreement"
    )
    parser.add_argument(
        "--model",
        default="Nanbeige/Nanbeige4.2-3B",
        help="Base Hugging Face model name or path",
    )
    parser.add_argument(
        "--weights",
        default="mtp_output/nanbeige_mtp_head.pt",
        help="Path to the trained MTP PyTorch state dict",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    evaluate_acceptance_rate(
        base_model_name=args.model,
        mtp_weights_path=args.weights,
        device_name=args.device,
    )
