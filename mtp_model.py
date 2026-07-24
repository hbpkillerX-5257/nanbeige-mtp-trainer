import torch
import torch.nn as nn
import torch.nn.functional as F


class MTPModule(nn.Module):
    """
    Multi-Token Prediction Layer for Nanbeige 4.2.
    Takes base model hidden state h_t and token embedding e(y_t),
    and refines representations for predicting token y_{t+1}.
    """
    def __init__(self, hidden_size: int, num_heads: int = 8, ffn_dim: int = 10752):
        super().__init__()
        # Projection layer: concatenates [h_t, e(y_t)]
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        
        # Transformer Layer
        self.attn_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=1e-5)
        self.ffn_gate = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.ffn_up   = nn.Linear(hidden_size, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, hidden_size, bias=False)

    def forward(self, h_t: torch.Tensor, emb_next: torch.Tensor) -> torch.Tensor:
        # Concatenate hidden state and embedding along feature dimension
        x = torch.cat([h_t, emb_next], dim=-1)
        x = self.eh_proj(x)
        
        # Attention block with residual connection
        norm_x = self.attn_norm(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # SwiGLU FFN block with residual connection
        norm_x2 = self.ffn_norm(x)
        swiglu = F.silu(self.ffn_gate(norm_x2)) * self.ffn_up(norm_x2)
        x = x + self.ffn_down(swiglu)
        return x

    def export_llama_cpp_state_dict(self) -> dict:
        """
        Export state dict using standard llama.cpp MTP tensor naming conventions.
        """
        state = self.state_dict()
        export_dict = {}
        for key, val in state.items():
            if key == "eh_proj.weight":
                export_dict["nextn.eh_proj.weight"] = val.cpu()
            elif key.startswith("attn_norm"):
                export_dict["mtp.0.attn_norm.weight"] = val.cpu()
            elif key.startswith("attn."):
                export_dict[f"mtp.0.{key}"] = val.cpu()
            elif key.startswith("ffn_norm"):
                export_dict["mtp.0.ffn_norm.weight"] = val.cpu()
            elif key.startswith("ffn_"):
                export_dict[f"mtp.0.{key}"] = val.cpu()
            else:
                export_dict[f"mtp.0.{key}"] = val.cpu()
        return export_dict
