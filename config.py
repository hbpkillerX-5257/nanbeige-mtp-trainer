from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingConfig:
    # Model Configuration
    base_model_name: str = "Nanbeige/Nanbeige4.2-3B"
    
    # Dataset Configuration (Default to Alpaca for Instruction/QnA models)
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_config: Optional[str] = None
    max_seq_len: int = 1024
    max_samples: int = 5000  # Limited dataset for testing
    
    # Checkpointing
    resume_from_checkpoint: bool = True
    checkpoint_dir: str = "./mtp_output"
    
    # MTP Architecture Parameters
    num_mtp_layers: int = 1
    num_heads: int = 8
    ffn_dim: int = 10752
    
    # Training Hyperparameters
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    epochs: int = 1
    mixed_precision: str = "fp16"  # "fp16" or "bf16"
    
    # Saving & Output
    output_dir: str = "./mtp_output"
    checkpoint_name: str = "nanbeige_mtp_head.pt"
    save_safetensors: bool = True
