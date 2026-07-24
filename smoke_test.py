import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def run_smoke_test():
    model_name = "Nanbeige/Nanbeige4.2-3B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"=== Starting Smoke Test on {device} ===")
    
    # 1. Tokenizer test
    print("\n--- Test 1: Tokenizer & Format ---")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    text = "The history of artificial intelligence began in antiquity, with"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    print(f"Raw text: {text}")
    print(f"Tokenized IDs: {inputs['input_ids'][0].tolist()}")
    for i, tok_id in enumerate(inputs['input_ids'][0]):
        print(f"  Token {i}: {tok_id.item()} -> '{tokenizer.decode([tok_id.item()])}'")
        
    dtypes_to_test = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    
    # Test each precision to isolate the collapse
    for name, dtype in dtypes_to_test.items():
        print(f"\n\n--- Test: Base Model in {name} ---")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                torch_dtype=dtype, 
                trust_remote_code=True,
                device_map={"": device}
            )
            model.eval()
            
            with torch.no_grad():
                # Pass WITHOUT attention mask
                outputs_no_mask = model(inputs["input_ids"], output_hidden_states=True)
                logits = outputs_no_mask.logits
                hidden = outputs_no_mask.hidden_states[-1]
                
                print(f"[{name}] WITHOUT Attention Mask:")
                print(f"  Hidden States NaN? : {torch.isnan(hidden).any().item()}")
                print(f"  Logits NaN?        : {torch.isnan(logits).any().item()}")
                
                preds = torch.argmax(logits, dim=-1)[0]
                print(f"  Predictions: {[tokenizer.decode([p.item()]) for p in preds]}")

                # Pass WITH attention mask
                outputs_mask = model(inputs["input_ids"], attention_mask=inputs["attention_mask"], output_hidden_states=True)
                logits_m = outputs_mask.logits
                hidden_m = outputs_mask.hidden_states[-1]
                
                print(f"\n[{name}] WITH Attention Mask:")
                print(f"  Hidden States NaN? : {torch.isnan(hidden_m).any().item()}")
                print(f"  Logits NaN?        : {torch.isnan(logits_m).any().item()}")
                
                preds_m = torch.argmax(logits_m, dim=-1)[0]
                print(f"  Predictions: {[tokenizer.decode([p.item()]) for p in preds_m]}")

            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"[{name}] FAILED to load or run: {e}")

if __name__ == "__main__":
    run_smoke_test()
