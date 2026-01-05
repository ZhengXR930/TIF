# TIF: Learning Temporal Invariance in Android Malware Detectors

This repository contains the implementation for TIF (Two-stage Information Filtering), a method for learning temporally invariant features in Android malware detection. The codebase includes TIF and baseline methods for comparison.

## Overview

TIF consists of two stages:
- **Stage 1**: Discriminative Information Amplification - enhances discriminative features across environments
- **Stage 2**: Unstable Information Suppression - suppresses temporally unstable features

The repository also includes baseline methods: SVM, DeepDrebin, T-Stability, and MPC (Multi-Proxy Contrastive loss).

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd SaTML
```

2. Install dependencies:
```bash
pip install torch numpy scikit-learn scipy tqdm pandas
```

## Quick Start

### Using the Run Script (Recommended)

Edit the variables at the top of `run_experiment.sh` and run:

```bash
./run_experiment.sh
```

Example configuration in `run_experiment.sh`:
```bash
METHOD=tif
MODE=stage1
SEED=1
DATA_FOLDER=/path/to/data
RESULT_FOLDER=/path/to/results
SAVE_FOLDER=/path/to/checkpoints
BATCH_SIZE=256
LEARNING_RATE=0.0001
EPOCHS=100
```

The script will:
- Run the experiment in the background with `nohup`
- Save logs to `logs/` directory with timestamp
- Display the process ID for monitoring

Monitor the log:
```bash
tail -f logs/tif_stage1_seed1_YYYYMMDD_HHMMSS.log
```

### Download Artifact
[Download Here](https://drive.google.com/drive/folders/1xLXi_9En2yHRrybY_SL0RoTvwP3Q_iR1?usp=sharing)

### Quick Test
+ Download artifact
+ load trained model `stage1_model.pt` and `tif_model.pt` for params: 
+ set `PROCESSED_DATA_FOLDER` as the path of `processed_data.zip`
+ set `RESULT_FOLDER`
+ run `run_experiment.sh`


### Using Command Line Interface

The main script `main.py` supports command-line arguments:

#### TIF Method

**Stage 1 (Discriminative Information Amplification):**
```bash
python main.py --method tif --mode stage1 \
    --data_folder /path/to/data \
    --result_folder /path/to/results \
    --save_folder /path/to/checkpoints \
    --seed 1 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --epochs 100
```

**Stage 2 (Unstable Information Suppression):**
```bash
python main.py --method tif --mode stage2 \
    --data_folder /path/to/data \
    --result_folder /path/to/results \
    --save_folder /path/to/checkpoints \
    --seed 1 \
    --batch_size 512 \
    --learning_rate 0.0001 \
    --penalty_weight 0.05 \
    --epochs 100
```

**TIF (Full Pipeline - loads stage1 model):**
```bash
python main.py --method tif --mode tif \
    --data_folder /path/to/data \
    --result_folder /path/to/results \
    --save_folder /path/to/checkpoints \
    --best_stg1_model_path /path/to/stage1_model.pt \
    --seed 1 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --epochs 20
```

**MPC (Multi-Proxy Contrastive Loss - tests MPC loss solely):**
```bash
python main.py --method tif --mode mpc \
    --data_folder /path/to/data \
    --result_folder /path/to/results \
    --save_folder /path/to/checkpoints \
    --seed 1 \
    --batch_size 256 \
    --learning_rate 0.0001 \
    --con_loss_weight 1.0 \
    --epochs 60
```

#### Baseline Methods (Drebin)

**SVM Baseline:**
```bash
python main.py --method drebin --mode svm \
    --data_folder /path/to/data \
    --result_folder /path/to/results
```

**DeepDrebin:**
```bash
python main.py --method drebin --mode deep \
    --data_folder /path/to/data \
    --result_folder /path/to/results \
    --save_folder /path/to/checkpoints \
    --seed 1 \
    --batch_size 128 \
    --learning_rate 0.0001 \
    --epochs 30
```

**T-Stability:**
```bash
python main.py --method drebin --mode ts \
    --data_folder /path/to/data \
    --result_folder /path/to/results \
    --save_folder /path/to/checkpoints \
    --seed 1
```

## Arguments

### Required Arguments
- `--method`: Method to use (`tif` or `drebin`)
- `--mode`: Mode to run
  - For `tif`: `stage1`, `stage2`, `tif`, or `mpc` (MPC tests the contrastive loss solely)
  - For `drebin`: `svm`, `deep`, or `ts`

### Path Arguments
- `--data_folder`: Path to data folder (default: `/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features_old/`)
- `--result_folder`: Path to result folder (default: `/cs/academic/phd3/xinrzhen/xinran/SaTML/result_rf`)
- `--save_folder`: Path to save checkpoints (default: `/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt`)

### Model Arguments
- `--best_model_path`: Path to best model checkpoint (optional)
- `--best_stg1_model_path`: Path to best stage 1 model checkpoint (required for `tif` mode)
- `--best_stg2_model_path`: Path to best stage 2 model checkpoint (optional for `tif` mode)

### Training Arguments
- `--seed`: Random seed for reproducibility (default: 1)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--batch_size`: Batch size for training (default: 256)
- `--eval_batch_size`: Batch size for evaluation (default: 128)
- `--learning_rate`: Learning rate (default: 0.0001)
- `--epochs`: Number of training epochs (default: 100)

### TIF-Specific Arguments
- `--con_loss_weight`: Contrastive loss weight (default: 1.0)
- `--penalty_weight`: Penalty weight for stage 2 (default: 0.05)
- `--mpc_load_mode`: MPC loading mode (`full`, `proj_only`, or `none`, default: `full`)
- `--weight_decay`: Weight decay for optimizer (default: 0)

### Data Arguments
- `--test_list`: Comma-separated list of test months or path to file with one month per line (optional, uses default list if not provided)

## Code Structure

### Core Files
- `main.py`: Main entry point with command-line interface
- `stage1_trainer.py`: Stage 1 trainer (Discriminative Information Amplification)
- `stage2_trainer.py`: Stage 2 trainer (Unstable Information Suppression)
- `trainer.py`: Base trainer for baseline methods
- `model.py`: Model definitions (DrebinMLP, DrebinMLP_IRM)
- `loss_mpc.py`: MPC (Multi-Proxy Contrastive) loss implementation
- `utils.py`: Utility functions for data loading and processing

### Baseline and Utilities
- `base_line/drebin.py`: Baseline methods (SVM, etc.)
- `t_stability.py`: T-Stability implementation
- `tesseract/`: Tesseract metrics and evaluation utilities (AUT, etc.)
- `create_dataset.py`: Dataset creation and preprocessing

### Scripts
- `run_experiment.sh`: Shell script for running experiments with nohup (edit variables at top)

## Dataset

The code expects data in the following format:
- Training data: `{data_folder}/train_data.pkl`
- Validation data: `{data_folder}/val_data.pkl`
- Test data: `{data_folder}/{month}.pkl` for each test month (e.g., `2015-01.pkl`, `2015-02.pkl`, ...)

Use `create_dataset.py` to preprocess and create datasets from raw features.


## Citation

If you use this code, please cite the corresponding paper.
```
@article{zheng2025learning,
  title={Learning temporal invariance in android malware detectors},
  author={Zheng, Xinran and Yang, Shuo and Ngai, Edith CH and Jana, Suman and Cavallaro, Lorenzo},
  journal={arXiv preprint arXiv:2502.05098},
  year={2025}
}
```
