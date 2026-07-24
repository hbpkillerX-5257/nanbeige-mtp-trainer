import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from mtp_model import MTPModule
from config import TrainingConfig
from tqdm import tqdm

def evaluate_acceptance_rate(
    base_model_name="Nanbeige/Nanbeige4.2-3B",
    mtp_weights_path="mtp_output/nanbeige_mtp_head.pt",
    text_sample="The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. As the field evolved, researchers developed mathematical models of human reasoning.",
    device="cuda:0"
):
    print("=== Loading Tokenizer and Base Model ===")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Patch rope scaling for transformers compatibility if needed
    model_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
    if hasattr(model_config, "rope_scaling") and model_config.rope_scaling is not None:
        if isinstance(model_config.rope_scaling, dict):
            rope_type = model_config.rope_scaling.get("type", model_config.rope_scaling.get("rope_type", None))
            if rope_type is None or rope_type == "default":
                model_config.rope_scaling = None
            else:
                model_config.rope_scaling.setdefault("type", rope_type)
                model_config.rope_scaling.setdefault("factor", 1.0)
                
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, 
        config=model_config,
        torch_dtype=torch.float16, 
        trust_remote_code=True,
        device_map={"": device}
    )
    base_model.eval()

    print(f"=== Loading MTP Head from {mtp_weights_path} ===")
    config = TrainingConfig()
    mtp_module = MTPModule(
        hidden_size=base_model.config.hidden_size,
        num_heads=config.num_heads,
        ffn_dim=config.ffn_dim
    ).to(device=device, dtype=torch.float32)
    
    # Load PyTorch weights
    state_dict = torch.load(mtp_weights_path, map_location=device, weights_only=True)
    mtp_module.load_state_dict(state_dict)
    mtp_module.eval()

    print("=== Running Evaluation ===")
    inputs = tokenizer(text_sample, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        # Get base model outputs
        outputs = base_model(input_ids, output_hidden_states=True, use_cache=False)
        
        # Base model predictions for next token
        # logits shape: [1, seq_len, vocab_size]
        base_logits = outputs.logits
        base_preds = torch.argmax(base_logits, dim=-1) # [1, seq_len]
        
        # Hidden states h_t (for step t): [1, seq_len-2, D] 
        h_t = outputs.hidden_states[-1][:, :-2, :]
        
        # Token embeddings e(y_{t+1}): [1, seq_len-2, D]
        embed_layer = base_model.get_input_embeddings()
        
        # In speculative decoding, we use the base model's prediction as the proposed y_{t+1}
        # But for exact accuracy, we use the ground truth token at t+1 just like training.
        # This isolates MTP's accuracy without compounding base model errors.
        emb_next = embed_layer(input_ids[:, 1:-1])
        
        # Forward pass through MTP module
        mtp_features = mtp_module(h_t.to(torch.float32), emb_next.to(torch.float32))
        
        # Compute MTP logits
        lm_head = base_model.get_output_embeddings()
        mtp_logits = F.linear(mtp_features.to(torch.float16), lm_head.weight)
        mtp_preds = torch.argmax(mtp_logits, dim=-1) # [1, seq_len-2]
        
        # Alignment:
        # Base model prediction for step t+2 is base_preds[:, 1:-1] (since base_preds[i] predicts t+i+1)
        # MTP prediction for step t+2 is mtp_preds
        target_base_preds = base_preds[:, 1:-1]
        
        correct = (mtp_preds == target_base_preds).sum().item()
        total = target_base_preds.numel()
        
        acceptance_rate = (correct / total) * 100
        
        print("\n" + "="*50)
        print(f"Total tokens evaluated: {total}")
        print(f"Tokens matching base model (Accepted): {correct}")
        print(f"Top-1 Acceptance Rate: {acceptance_rate:.2f}%")
        print("="*50)
        
        # Print a small sample
        print("\nSample Predictions:")
        for i in range(min(10, total)):
            context = tokenizer.decode(input_ids[0, :i+2])
            base_tok = tokenizer.decode([target_base_preds[0, i].item()])
            mtp_tok = tokenizer.decode([mtp_preds[0, i].item()])
            match = "✅" if base_tok == mtp_tok else "❌"
            print(f"Context: {repr(context)}")
            print(f"  Base Model Predicts: {repr(base_tok)}")
            print(f"  MTP Head Predicts  : {repr(mtp_tok)} {match}\n")

if __name__ == "__main__":
    evaluate_acceptance_rate()
