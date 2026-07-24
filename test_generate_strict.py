import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def run():
    model_name = "Nanbeige/Nanbeige4.2-3B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    print("Loading model in fp32...")
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        if isinstance(model_config.rope_scaling, dict):
            rope_type = model_config.rope_scaling.get("type", model_config.rope_scaling.get("rope_type", None))
            if rope_type is None or rope_type == "default":
                model_config.rope_scaling = None
            else:
                model_config.rope_scaling.setdefault("type", rope_type)
                model_config.rope_scaling.setdefault("factor", 1.0)
                
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        config=model_config,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation="eager"
    )
    model.eval()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about the history of artificial intelligence."}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\nTesting with formatted prompt:\n{repr(text)}")
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True, use_cache=False)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)[0]
        
        hidden = outputs.hidden_states[-1][0] # shape: [seq_len, hidden_size]
        
        print("\n--- Diagnostic Analysis ---")
        print(f"Logits shape: {logits.shape}")
        
        # Check weights
        embed_w = model.model.embed_tokens.weight
        lm_head_w = model.lm_head.weight
        print(f"\nEmbed weight mean: {embed_w.mean().item():.5f}, std: {embed_w.std().item():.5f}")
        print(f"lm_head weight mean: {lm_head_w.mean().item():.5f}, std: {lm_head_w.std().item():.5f}")
        print(f"Are they tied? {embed_w.data_ptr() == lm_head_w.data_ptr()}")
        
        # Check if hidden states change across positions
        h_diffs = []
        for i in range(1, hidden.shape[0]):
            diff = torch.norm(hidden[i] - hidden[i-1]).item()
            h_diffs.append(diff)
        
        print(f"\nHidden state differences between consecutive tokens:")
        print(f"Diffs: {[round(d, 4) for d in h_diffs]}")
        
        # Check if all tokens predict the same thing
        unique_preds = torch.unique(preds)
        print(f"\nUnique tokens predicted across {len(preds)} positions: {len(unique_preds)}")
        
        for i, p in enumerate(preds):
            ctx = tokenizer.decode(inputs["input_ids"][0][:i+1])
            pred = tokenizer.decode([p.item()])
            print(f"Token {i} | Context: {repr(ctx)} | Predicts: {repr(pred)}")

if __name__ == "__main__":
    run()
