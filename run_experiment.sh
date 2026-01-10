#!/bin/bash

# Set parameters
METHOD=tif
MODE=tif  
SEED=1
PROCESSED_DATA_FOLDER='/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features_new'
RESULT_FOLDER='/cs/academic/phd3/xinrzhen/xinran/SaTML/result_rf'
SAVE_FOLDER='/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/new_features'
DEVICE=cuda
EVAL_BATCH_SIZE=128

# Stage 1 parameters
STAGE1_BATCH_SIZE=512
STAGE1_LEARNING_RATE=0.0001
STAGE1_CON_LOSS_WEIGHT=1.0
STAGE1_WEIGHT_DECAY=1e-4
STAGE1_EPOCHS=40
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
            --eval_batch_size ${EVAL_BATCH_SIZE} \
            --stage1_batch_size ${STAGE1_BATCH_SIZE} \
            --stage2_batch_size ${STAGE2_BATCH_SIZE} \
            --stage1_learning_rate ${STAGE1_LEARNING_RATE} \
            --stage2_learning_rate ${STAGE2_LEARNING_RATE} \
            --stage1_con_loss_weight ${STAGE1_CON_LOSS_WEIGHT} \
            --stage2_con_loss_weight ${STAGE2_CON_LOSS_WEIGHT} \
            --stage1_weight_decay ${STAGE1_WEIGHT_DECAY} \
            --stage2_weight_decay ${STAGE2_WEIGHT_DECAY} \
            --stage1_epochs ${STAGE1_EPOCHS} \
            --stage2_epochs ${STAGE2_EPOCHS} \
            --stage1_n_proxy ${STAGE1_N_PROXY} \
            --stage2_n_proxy ${STAGE2_N_PROXY} \
            --stage1_early_stop_patience ${STAGE1_EARLY_STOP_PATIENCE} \
            --stage2_early_stop_patience ${STAGE2_EARLY_STOP_PATIENCE} \
            --penalty_weight ${PENALTY_WEIGHT} \
            --mpc_load_mode ${MPC_LOAD_MODE} \
            >> ${LOG_FILE} 2>&1 &

echo "Process started, PID: $!"
echo "Log file: ${LOG_FILE}"
echo "To monitor: tail -f ${LOG_FILE}"
echo "To stop: kill $!"

