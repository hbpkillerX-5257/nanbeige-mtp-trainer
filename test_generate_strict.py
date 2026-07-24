import sys
import types

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

# 3. NOW LOAD YOUR MODEL
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache

# Patch DynamicCache for newer transformers compatibility
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: getattr(self, "get_seq_length", lambda: None)()

def run():
    model_id = "Nanbeige/Nanbeige4.2-3B"

    # Fix RoPE scaling dict schema
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, "rope_scaling") and config.rope_scaling:
        if isinstance(config.rope_scaling, dict):
            config.rope_scaling["type"] = "linear"
            config.rope_scaling["factor"] = 1.0

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        device_map="cuda"
    )

    # 4. RUN GENERATION
    messages = [{"role": "user", "content": "Which number is bigger, 9.11 or 9.8?"}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")

    output_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        eos_token_id=tokenizer.eos_token_id or 166101
    )

    response = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    run()
