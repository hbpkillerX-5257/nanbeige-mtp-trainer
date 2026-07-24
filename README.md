# Nanbeige 4.2 3B Multi-Token Prediction (MTP) Trainer

A lightweight PyTorch repository for training a **Multi-Token Prediction (MTP)** head on top of Nanbeige 4.2 3B using Distributed Data Parallel (**DDP** / **`torchrun`**) across **2x NVIDIA T4 GPUs** (Kaggle or local).

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

## 🚀 How to Run on Kaggle (2x Tesla T4 GPUs)

### Step 1: Copy Package to Kaggle Notebook
Upload or clone `nanbeige_mtp_trainer` into your Kaggle Notebook directory.

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

---

## 📦 Model Saving & Output Formats

The training script automatically saves the model in three formats under `./mtp_output/`:

1. **`nanbeige_mtp_head.pt`**: Standard PyTorch state dict.
2. **`nanbeige_mtp_head.safetensors`**: HuggingFace Safetensors format.
3. **`nanbeige_mtp_gguf_tensors.pt`**: llama.cpp-formatted tensor dictionary with `mtp.0.*` and `nextn.eh_proj` keys.

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
