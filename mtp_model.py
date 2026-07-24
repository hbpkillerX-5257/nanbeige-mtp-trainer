import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPModule(nn.Module):
    """
    Multi-Token Prediction Layer for Nanbeige 4.2.
    Takes base model hidden state h_t and token embedding e(y_t),
    and refines representations for predicting token y_{t+1}.
    """
    def __init__(self, hidden_size: int, base_layer: nn.Module = None):
        super().__init__()
        # Projection layer: concatenates [h_t, e(y_t)]
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # Clone the exact base layer to perfectly preserve GQA, RoPE, and MLP architectures!
        self.transformer_block = copy.deepcopy(base_layer) if base_layer is not None else None

        self._init_weights()

    def _init_weights(self):
        # We only initialize our custom projection layer randomly
        nn.init.normal_(self.eh_proj.weight, mean=0.0, std=0.02)
        # We DO NOT re-initialize transformer_block because we want the pre-trained weights!

    def forward(self, h_t: torch.Tensor, emb_next: torch.Tensor, position_ids=None) -> torch.Tensor:
        # Concatenate hidden state and embedding along feature dimension
        x = torch.cat([h_t, emb_next], dim=-1)
        x = self.eh_proj(x)
        
        if self.transformer_block is None:
            return x # Fallback if base_layer wasn't provided, though it will crash later
            
        B, S, D = x.shape
        
        # Build standard HF causal attention mask [B, 1, S, S]
        causal_mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        attention_mask = torch.zeros(B, 1, S, S, device=x.device, dtype=x.dtype)
        attention_mask.masked_fill_(causal_mask, torch.finfo(x.dtype).min)
        
        if position_ids is None:
            position_ids = torch.arange(S, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            
        # Forward pass through the cloned Nanbeige base layer
        # Nanbeige base layer returns a tuple: (hidden_states, ...)
        outputs = self.transformer_block(
            hidden_states=x, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False
        )
        return outputs[0]

    def export_llama_cpp_state_dict(self) -> dict:
        """
        Export state dict using standard llama.cpp MTP tensor naming conventions.
        Because we cloned a base layer, the internal names match the base model (e.g. self_attn.q_proj.weight).
        We map them to mtp.0.* for llama.cpp.
        """
        state = self.state_dict()
        export_dict = {}
        for key, val in state.items():
            if key == "eh_proj.weight":
                export_dict["nextn.eh_proj.weight"] = val.cpu()
            elif key.startswith("transformer_block.input_layernorm"):
                export_dict["mtp.0.attn_norm.weight"] = val.cpu()
            elif key.startswith("transformer_block.post_attention_layernorm"):
                export_dict["mtp.0.ffn_norm.weight"] = val.cpu()
            elif key.startswith("transformer_block.self_attn."):
                sub_name = key.split("self_attn.")[1] # e.g. q_proj.weight
                export_dict[f"mtp.0.attn.{sub_name}"] = val.cpu()
            elif key.startswith("transformer_block.mlp."):
                sub_name = key.split("mlp.")[1] # e.g. gate_proj.weight
                export_dict[f"mtp.0.ffn_{sub_name}"] = val.cpu()
        return export_dict
