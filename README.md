# TIF: Learning Temporal Invariance in Android Malware Detectors

This repository contains the implementation for TIF (Two-stage Information Filtering), a method for learning temporally invariant features in Android malware detection. The codebase includes TIF and baseline methods for comparison.

## Overview

TIF consists of two stages:
- **Stage 1**: Discriminative Information Amplification - enhances discriminative features across environments
- **Stage 2**: Unstable Information Suppression - suppresses temporally unstable features

The repository also includes baseline methods: SVM, DeepDrebin, T-Stability.

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd TIF
```

2. Install dependencies:
```bash
pip install torch numpy scikit-learn scipy tqdm pandas
```

## Quick Start


### Download Artifact
[Download Here](https://drive.google.com/drive/folders/1xLXi_9En2yHRrybY_SL0RoTvwP3Q_iR1?usp=sharing)

+ Download artifact
+ Set `PROCESSED_DATA_FOLDER` as the path of `processed_features.zip`
+ Set `RESULT_FOLDER` and `SAVE_FOLDER` in `run_experiment.sh`

### Using the Run Script

Edit the variables at the top of `run_experiment.sh` and run:

```bash
./run_experiment.sh
```

Example configuration in `run_experiment.sh`:
```bash
METHOD=tif
MODE=tif
SEED=1
PROCESSED_DATA_FOLDER=/path/to/data
RESULT_FOLDER=/path/to/results
SAVE_FOLDER=/path/to/checkpoints

# Stage 1 parameters
STAGE1_BATCH_SIZE=512
STAGE1_LEARNING_RATE=0.0001
STAGE1_CON_LOSS_WEIGHT=1.0
STAGE1_WEIGHT_DECAY=0
STAGE1_EPOCHS=30
STAGE1_N_PROXY=3
STAGE1_EARLY_STOP_PATIENCE=100

# Stage 2 parameters
STAGE2_BATCH_SIZE=1024
STAGE2_LEARNING_RATE=0.0001
STAGE2_CON_LOSS_WEIGHT=0.1
STAGE2_WEIGHT_DECAY=1e-3
STAGE2_EPOCHS=20
STAGE2_N_PROXY=3
STAGE2_EARLY_STOP_PATIENCE=5

# TIF-specific parameters
PENALTY_WEIGHT=1.0
MPC_LOAD_MODE=full
```

The script will:
- Run the experiment in the background with `nohup`
- Save logs to `logs/` directory with timestamp
- Display the process ID for monitoring

Monitor the log:
```bash
tail -f logs/tif_tif_seed1_YYYYMMDD_HHMMSS.log
```



### Using Command Line Interface

The main script `main.py` supports command-line arguments:

#### TIF Method

```bash
python main.py --method tif --mode tif \
    --data_folder /path/to/data \
    --result_folder /path/to/results \
    --save_folder /path/to/checkpoints \
    --seed 1 \
    --stage1_batch_size 512 \
    --stage2_batch_size 1024 \
    --stage1_learning_rate 0.0001 \
    --stage2_learning_rate 0.0001 \
    --stage1_con_loss_weight 1.0 \
    --stage2_con_loss_weight 0.1 \
    --stage1_weight_decay 0 \
    --stage2_weight_decay 1e-3 \
    --stage1_epochs 30 \
    --stage2_epochs 20 \
    --stage1_n_proxy 3 \
    --stage2_n_proxy 3 \
    --stage1_early_stop_patience 100 \
    --stage2_early_stop_patience 5 \
    --penalty_weight 1.0 \
    --mpc_load_mode full
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
  - For `tif`: `tif`
  - For `drebin`: `svm`, `deep`, or `ts`

### Path Arguments
- `--data_folder`: Path to data folder (must contain `train_data.pkl` and `val_data.pkl`)
- `--result_folder`: Path to result folder (for saving evaluation results)
- `--save_folder`: Path to save checkpoints

### Training Arguments
- `--seed`: Random seed for reproducibility (default: 1)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)
- `--eval_batch_size`: Batch size for evaluation (default: 128)

### TIF-Specific Arguments (Stage 1)
- `--stage1_batch_size`: Batch size for Stage 1 training (default: 512)
- `--stage1_learning_rate`: Learning rate for Stage 1 (default: 0.0001)
- `--stage1_con_loss_weight`: Contrastive loss weight for Stage 1 (default: 1.0)
- `--stage1_weight_decay`: Weight decay for Stage 1 (default: 0)
- `--stage1_epochs`: Number of epochs for Stage 1 (default: 30)
- `--stage1_n_proxy`: Number of proxies per class in Stage 1 (default: 3)
- `--stage1_early_stop_patience`: Early stopping patience for Stage 1 (default: 100)

### TIF-Specific Arguments (Stage 2)
- `--stage2_batch_size`: Batch size for Stage 2 training (default: 1024)
- `--stage2_learning_rate`: Learning rate for Stage 2 (default: 0.0001)
- `--stage2_con_loss_weight`: Contrastive loss weight for Stage 2 (default: 0.1)
- `--stage2_weight_decay`: Weight decay for Stage 2 (default: 1e-3)
- `--stage2_epochs`: Number of epochs for Stage 2 (default: 20)
- `--stage2_n_proxy`: Number of proxies per class in Stage 2 (default: 3)
- `--stage2_early_stop_patience`: Early stopping patience for Stage 2 (default: 5)
- `--penalty_weight`: IRM penalty weight for Stage 2 (default: 1.0)
- `--mpc_load_mode`: MPC loading mode (`full`, `proj_only`, `none`, or `auto`, default: `full`)

### Data Arguments
- `--test_list`: Comma-separated list of test months or path to file with one month per line (optional, uses default list if not provided)

## Code Structure

### Core Files
- `main.py`: Main entry point with command-line interface (supports end-to-end TIF training)
- `stage1_trainer.py`: Stage 1 trainer (Discriminative Information Amplification with multi-proxy contrastive learning)
- `stage2_trainer.py`: Stage 2 trainer (Unstable Information Suppression with IRM penalty)
- `trainer.py`: Base trainer for baseline methods
- `model.py`: Model definitions (DrebinMLP, DrebinMLP_IRM)
- `loss_mpc.py`: MPC (Multi-Proxy Contrastive) loss implementation
- `utils.py`: Utility functions for data loading and processing
- `create_dataset.py`: Process original dataset to generate train/val/test sets with configurable environment split modes

### Baseline and Utilities
- `base_line/drebin.py`: Baseline methods (SVM, etc.)
- `t_stability.py`: T-Stability implementation
- `tesseract/`: Tesseract metrics and evaluation utilities (AUT, etc.)

### Scripts
- `run_experiment.sh`: Shell script for running experiments with nohup (edit variables at top for Stage 1/2 parameters)

## Dataset

The code expects data in the following format:
- Training data: `{data_folder}/train_data.pkl`
- Validation data: `{data_folder}/val_data.pkl`
- Test data: `{data_folder}/{month}.pkl` for each test month (e.g., `2015-01.pkl`, `2015-02.pkl`, ...)

### Creating Datasets

Use `create_dataset.py` to preprocess and create datasets from raw features. The script supports three modes:

**Regenerate Mode** (extract features using different selector):
```bash
python create_dataset.py --mode regenerate \
    --method randomforest \
    --n_features 10000 \
    --batch_size 10000 \
    --env_split_mode quarter \
    --output_folder /path/to/output
```

**Predefine Mode** (use external feature list):
```bash
python create_dataset.py --mode predefine \
    --predefined_features_file /path/to/features.txt \
    --vectorizer_file /path/to/vectorizer.pkl \
    --env_split_mode month \
    --output_folder /path/to/output
```

**Process Mode** (process files with existing features):
```bash
python create_dataset.py --mode process \
    --feature_list_file /path/to/features.txt \
    --vectorizer_file /path/to/vectorizer.pkl \
    --input_files /path/to/file1.pkl /path/to/file2.pkl \
    --output_folder /path/to/output
```

#### Environment Split Modes

The `--env_split_mode` parameter controls how training samples are assigned to environments:

- `quarter`: Split by quarters (Q1=0, Q2=1, Q3=2, Q4=3)
- `month`: Split by months (Jan=0, Feb=1, ..., Dec=11)
- `uniform`: Uniformly divide samples into `n_envs` environments (use `--n_envs` to specify number, default: 4)

Example with uniform split:
```bash
python create_dataset.py --mode regenerate \
    --env_split_mode uniform \
    --n_envs 8 \
    --output_folder /path/to/output
```


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
