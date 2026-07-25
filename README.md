# Nanbeige 4.2 3B Multi-Token Prediction (MTP) Trainer

A lightweight PyTorch repository for training a **Multi-Token Prediction (MTP)** head on top of Nanbeige 4.2 3B using Distributed Data Parallel (**DDP** / **`torchrun`**).

> **BF16 hardware is required.** Nanbeige4.2 is published as a BF16 model and
> its teacher predictions collapse when forced to FP16 on NVIDIA T4 GPUs.
> Use BF16-capable GPUs such as A100, L4, A10, or newer. The trainer now fails
> before loading the model when BF16 is unavailable and also runs a teacher
> health check before the first training batch.

---

## 📁 Repository Structure

```
nanbeige_mtp_trainer/
├── __init__.py           # Package initialization
├── config.py             # Hyperparameters & configuration
├── mtp_model.py          # MTP Module Architecture & GGUF tensor exporter
├── dataset.py            # Tokenization & Distributed Data Sampler
├── train.py              # Main Distributed Training Script (PyTorch DDP)
├── export_mtp_gguf.py    # GGUF Merger Script for llama.cpp
├── run_torchrun.sh       # Multi-GPU launcher using torchrun
├── run_accelerate.sh     # Multi-GPU launcher using HuggingFace Accelerate
└── README.md             # Guide & documentation
```

---

## 🚀 How to Run

### Step 1: Copy Package to Kaggle Notebook
Upload or clone `nanbeige_mtp_trainer` into your notebook or working directory.

### Step 2: Run Multi-GPU Training (`torchrun`)

In a Kaggle code cell:

```bash
!chmod +x nanbeige_mtp_trainer/run_torchrun.sh
!./nanbeige_mtp_trainer/run_torchrun.sh
```

Or directly via `torchrun`:

```bash
!torchrun --nproc_per_node=2 nanbeige_mtp_trainer/train.py
```

To resume from the epoch checkpoint (including optimizer and scheduler state):

```bash
!torchrun --nproc_per_node=2 nanbeige_mtp_trainer/train.py --resume
```

Training does not resume unless `--resume` is supplied. The resumable state is
stored as `mtp_output/trainer_state.pt`; the standalone MTP weights remain in
`nanbeige_mtp_head.pt`.

The KL-distillation calculation streams over vocabulary chunks. LM-head
projections use the model dtype, while normalization, KL accumulation, and
gradient accumulation use FP32. This keeps the exact teacher distribution while
avoiding full `[tokens, vocabulary]` probability tensors.
`kd_vocab_chunk_size` in `config.py` controls the memory/throughput tradeoff.

For instruction datasets such as Alpaca, loss is computed only for target
tokens in the assistant answer. System and user prompt tokens provide context
but do not contribute to the distillation loss. Plain-text datasets continue to
train on all non-padding tokens.

---

## 📦 Model Saving & Output Formats

The training script automatically saves the model in three formats under `./mtp_output/`:

1. **`nanbeige_mtp_head.pt`**: Standard PyTorch state dict.
2. **`nanbeige_mtp_head.safetensors`**: HuggingFace Safetensors format.
3. **`nanbeige_mtp_gguf_tensors.pt`**: llama.cpp-formatted tensor dictionary with `mtp.0.*` and `nextn.eh_proj` keys.

The trainer also writes **`trainer_state.pt`** after each completed epoch for
reliable `--resume` operation.

Evaluate teacher-forced two-token agreement with the same model and tokenizer
settings used during training:

```bash
python3 nanbeige_mtp_trainer/eval.py \
    --weights ./mtp_output/nanbeige_mtp_head.pt
```

---

## 🛠️ Exporting to GGUF for llama.cpp (`draft-mtp`)

Once training completes, merge the trained MTP head into your base Nanbeige GGUF model:

```bash
python3 nanbeige_mtp_trainer/export_mtp_gguf.py \
    --base-gguf ../models/nanbeige4.2-3b-Q4_0.gguf \
    --mtp-weights ./mtp_output/nanbeige_mtp_gguf_tensors.pt \
    --output ../models/nanbeige4.2-3b-mtp-Q4_0.gguf
```

### Run MTP Speculative Inference in llama.cpp

```bash
./llama-cli -m models/nanbeige4.2-3b-mtp-Q4_0.gguf \
    --spec-type draft-mtp --spec-draft-n-max 1 \
    -ngl 99 -fa on -c 32768 -p "Your prompt"
```
