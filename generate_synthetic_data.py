import sys
import os
from pathlib import Path
import json
import torch
from tqdm import tqdm

# 1. UNINSTALL BROKEN FLASH_ATTN FROM PYTHON MEMORY
for mod_name in list(sys.modules.keys()):
    if mod_name.startswith("flash_attn"):
        del sys.modules[mod_name]
sys.modules["flash_attn"] = None

# 2. DISABLE TRANSFORMERS CHECKS
import transformers.utils.import_utils
transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
import transformers.dynamic_module_utils
transformers.dynamic_module_utils.check_imports = lambda filename: []

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from train import TrainingConfig

# Patch DynamicCache for newer transformers compatibility
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: getattr(self, "get_seq_length", lambda: None)()


def main():
    config = TrainingConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Base Model: {config.base_model_name}...")
    
    # Fix RoPE scaling dict schema
    model_config = AutoConfig.from_pretrained(config.base_model_name, trust_remote_code=True)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling:
        if isinstance(model_config.rope_scaling, dict):
            model_config.rope_scaling["type"] = "linear"
            model_config.rope_scaling["factor"] = 1.0

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        config=model_config,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda"
    )
    model.eval()

    print(f"Loading dataset: {config.dataset_name}...")
    try:
        if config.dataset_config:
            raw_dataset = load_dataset(config.dataset_name, config.dataset_config, split=f"train[:{config.max_samples}]")
        else:
            raw_dataset = load_dataset(config.dataset_name, split=f"train[:{config.max_samples}]")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    is_instruct = "instruction" in raw_dataset.column_names
    
    output_file = "synthetic_on_policy_data.jsonl"
    print(f"Generating 16 responses per prompt. Saving to {output_file}...")

    with open(output_file, "w") as f, torch.no_grad():
        for row in tqdm(raw_dataset):
            if is_instruct:
                prompt_text = f"{row['instruction']}\n{row.get('input', '')}".strip()
            else:
                prompt_text = row["text"]

            messages = [{"role": "user", "content": prompt_text}]
            
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"User: {prompt_text}\n\nAssistant: "
                
            input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)

            # Generate 16 responses
            try:
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    num_return_sequences=16,
                    eos_token_id=tokenizer.eos_token_id
                )
            except Exception as e:
                print(f"Skipping prompt due to generation error: {e}")
                continue

            responses = []
            for i in range(16):
                # Decode only the newly generated tokens
                generated_tokens = outputs[i][len(input_ids[0]):]
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response)

            # Save to JSONL
            record = {
                "prompt": prompt,
                "responses": responses
            }
            f.write(json.dumps(record) + "\n")
            
    print("Done! You can now use this dataset for MTP training.")

if __name__ == "__main__":
    main()
