from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # Model Configuration
    base_model_name: str = "Nanbeige/Nanbeige4.2-3B"
    model_revision: str = "a8a131e1689a819fb9119f16bf8a2629d09fd41c"
    
    # Dataset Configuration (Default to Alpaca for Instruction/QnA models)
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_config: Optional[str] = None
    max_seq_len: int = 1024
    max_samples: int = 52000  # Full Alpaca dataset
    
    # Checkpointing
    resume_from_checkpoint: bool = False
    checkpoint_dir: str = "./mtp_output"
    trainer_state_name: str = "trainer_state.pt"
    
    # Training Hyperparameters
    batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    epochs: int = 1
    # Nanbeige4.2 is a BF16 model. FP16 collapses on T4-class hardware.
    mixed_precision: str = "bf16"  # "bf16" or "fp32" recommended
    kd_vocab_chunk_size: int = 8192
    
    # Saving & Output
    output_dir: str = "./mtp_output"
    checkpoint_name: str = "nanbeige_mtp_head.pt"
    save_safetensors: bool = True
