#!/bin/bash

# Set parameters
METHOD=tif
MODE=tif  # Options: stage1, stage2, tif
SEED=1
PROCESSED_DATA_FOLDER=''
RESULT_FOLDER=''
SAVE_FOLDER=''
DEVICE=cuda
BATCH_SIZE=512  # Default: 512 for stage1, 1024 for stage2/tif
EVAL_BATCH_SIZE=128
LEARNING_RATE=0.0001
EPOCHS=30  # Default: 30 for stage1, 20 for stage2/tif
CON_LOSS_WEIGHT=1.0  # Default: 1.0 for stage1, 0.1 for stage2/tif
PENALTY_WEIGHT=1.0  # Default: 1.0 for stage2/tif
MPC_LOAD_MODE=full
WEIGHT_DECAY=0  # Default: 1e-4 for stage1, 1e-3 for stage2/tif
USE_MULTI_PROXY=true  # Default: true for stage1
STAGE1_N_PROXY=3
STAGE2_N_PROXY=3
EARLY_STOP_PATIENCE=100  # Default: 100 for stage1, 5 for stage2/tif
BEST_MODEL_PATH=""
BEST_STG1_MODEL_PATH="stage1_model.pt"
BEST_STG2_MODEL_PATH=""

TS=$(date "+%Y%m%d_%H%M%S")
LOG_FILE=logs/${METHOD}_${MODE}_seed${SEED}_${TS}.log

mkdir -p logs

nohup python -u main.py \
            --method ${METHOD} \
            --mode ${MODE} \
            --seed ${SEED} \
            --data_folder ${PROCESSED_DATA_FOLDER} \
            --result_folder ${RESULT_FOLDER} \
            --save_folder ${SAVE_FOLDER} \
            --device ${DEVICE} \
            --batch_size ${BATCH_SIZE} \
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --learning_rate ${LEARNING_RATE} \
            --epochs ${EPOCHS} \
            --con_loss_weight ${CON_LOSS_WEIGHT} \
            --penalty_weight ${PENALTY_WEIGHT} \
            --mpc_load_mode ${MPC_LOAD_MODE} \
            --weight_decay ${WEIGHT_DECAY} \
            ${USE_MULTI_PROXY:+--use_multi_proxy} \
            --stage1_n_proxy ${STAGE1_N_PROXY} \
            --stage2_n_proxy ${STAGE2_N_PROXY} \
            --early_stop_patience ${EARLY_STOP_PATIENCE} \
            ${BEST_MODEL_PATH:+--best_model_path ${BEST_MODEL_PATH}} \
            ${BEST_STG1_MODEL_PATH:+--best_stg1_model_path ${BEST_STG1_MODEL_PATH}} \
            ${BEST_STG2_MODEL_PATH:+--best_stg2_model_path ${BEST_STG2_MODEL_PATH}} \
            >> ${LOG_FILE} 2>&1 &

echo "Process started, PID: $!"
echo "Log file: ${LOG_FILE}"
echo "To monitor: tail -f ${LOG_FILE}"
echo "To stop: kill $!"

