# TABLeT: Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?

Official implementation of **Can Natural Image Autoencoders Compactly Tokenize fMRI Volumes for Long-Range Dynamics Modeling?** (CVPR 2026).

## Overview

TABLeT is a brain foundation model for general-purpose fMRI analysis. It tokenizes 3D fMRI volumes using a pretrained image tokenizer (DC-AE) and processes the resulting spatiotemporal token sequences with a Grouped Query Attention (GQA) Transformer.

**Supported downstream tasks:**
- **Regression:** Age prediction, intelligence prediction
- **Classification:** Gender, ADHD, HBN (movie classification)

**Supported datasets:** HCP, UKB, ADHD, HBN

## Setup

### Requirements

- Python >= 3.10
- CUDA >= 12.1

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install accelerate==1.5.2 scikit-learn pandas wandb omegaconf einops nilearn deepspeed==0.15.4 numpy==1.26.4
pip install monai --no-deps
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Data Preparation

### 1. Tokenize fMRI volumes with DC-AE

Convert raw fMRI volumes into latent tokens using the pretrained DC-AE encoder:

```bash
python dcae_tokenize.py \
    --img_dir /path/to/fMRI_frames \
    --latent_save_dir /path/to/save/tokens \
    --dcae_model_path /path/to/dc-ae-f32c32-in-1.0-diffusers \
    --axis 0 \
    --iter_idx 0 \
    --max_iter_idx 4
```

Repeat for `--axis 1` and `--axis 2` to get multi-axis tokenizations.

### 2. Organize data directory

```
/path/to/HCP_data/
├── latents_MNI152/
│   ├── axis0/
│   │   └── <subject_id>/
│   │       ├── frame_0.npy
│   │       ├── frame_1.npy
│   │       └── ...
│   ├── axis1/
│   └── axis2/
└── metadata/
    ├── HCP_metadata.csv
    └── split_by_gender_age_intelligence.pt
```

## Training

### Single run

```bash
accelerate launch --num_processes 1 train.py \
    --config configs/hcp_configs/tablet_hcp_intelligence_T256.yaml \
    --num-axis 3 \
    --test-set-id 0 \
    --lr 1e-4 \
    --batch-size 4 \
    --epochs 20 \
    --wandb-name tablet_hcp_intelligence
```

### Learning rate sweep (multi-GPU)

```bash
bash run.sh
```

This launches parallel experiments across 4 GPUs with learning rates `{1e-6, 1e-5, 1e-4, 1e-3}`.

### Key arguments

| Argument | Description |
|---|---|
| `--config` | Path to YAML config file |
| `--num-axis` | Number of slice axes (1 or 3) |
| `--test-set-id` | train/val/test split idx (0-3) |
| `--lr` | Learning rate |
| `--batch-size` | Batch size per GPU |
| `--epochs` | Number of training epochs |
| `--tablet_pretrained_weight_path` | Path to pretrained TABLeT weights for fine-tuning |
| `--checkpoint_dir` | Directory for saving checkpoints |
| `--resume_checkpoint_dir` | Path to checkpoint for resuming training |

### Configuration

Edit YAML config files in `configs/` to change:
- Dataset and task (`data.dataset`, `data.task`)
- Data type (`data.data_type`): `token`, `volume`, `volume_tff_pretrain`, `functional_connectivity`, `roi_signals`
- Model architecture (`model.name`, `model.params`)
- Training settings (`train.mixed_precision`, `train.use_warmup`)

## Project Structure

```
TABLeT/
├── train.py                    # Main training entry point
├── dcae_tokenize.py            # DC-AE tokenization script
├── run.sh                      # Multi-GPU experiment launcher
├── configs/                    # YAML configuration files
│   └── hcp_configs/
├── src/
│   ├── models/
│   │   ├── model_factory.py              # Model registry
│   │   ├── TABLeT.py                     # TABLeT (main model)
│   │   ├── swift.py                      # SwiFT baseline (Swin Transformer 4D)
│   │   └── ...                           # Other baseline models
│   ├── datasets/
│   │   ├── base_dataset.py      # Base dataset class
│   │   ├── hcp.py               # HCP dataset
│   │   ├── ukb.py               # UK Biobank dataset
│   │   ├── adhd.py              # ADHD dataset
│   │   └── hbn_movie_clf.py     # HBN movie classification dataset
│   └── trainers/
│       ├── generic_trainer.py   # Main trainer with train/val/test loops
│       └── ...                  # Specialized trainers (TFF phases, SwiFT)
└── README.md
```

## Citation

If you find this work useful, please cite:


## License

TBD
