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
        torch_dtype=torch.float32, 
        trust_remote_code=True,
        device_map={"": device}
    )
    model.eval()

    text = "The history of artificial intelligence began in antiquity, with"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], use_cache=False)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)[0]
        
        print(f"Logits shape: {logits.shape}")
        
        for i, p in enumerate(preds):
            ctx = tokenizer.decode(inputs["input_ids"][0][:i+1])
            pred = tokenizer.decode([p.item()])
            print(f"Token {i} | Context: {repr(ctx)} | Predicts: {repr(pred)}")
            
        print("\nLet's try model.generate() to see if it predicts something normal:")
        gen = model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print("Generated:", tokenizer.decode(gen[0]))

if __name__ == "__main__":
    run()
