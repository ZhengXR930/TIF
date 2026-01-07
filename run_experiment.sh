#!/bin/bash

# Set parameters
METHOD=tif
MODE=tif
SEED=1
PROCESSED_DATA_FOLDER='/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/dataset/processed_features_old/'
RESULT_FOLDER='/cs/academic/phd3/xinrzhen/xinran/SaTML/result_rf'
SAVE_FOLDER='/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt'
DEVICE=cuda
BATCH_SIZE=512
EVAL_BATCH_SIZE=128
LEARNING_RATE=0.0001
EPOCHS=10
CON_LOSS_WEIGHT=1.0
PENALTY_WEIGHT=0.05
MPC_LOAD_MODE=full
WEIGHT_DECAY=0
BEST_MODEL_PATH=""
BEST_STG1_MODEL_PATH="/scratch_NOT_BACKED_UP/NOT_BACKED_UP/xinran/ckpt/stage1_model.pt"
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
            ${BEST_MODEL_PATH:+--best_model_path ${BEST_MODEL_PATH}} \
            ${BEST_STG1_MODEL_PATH:+--best_stg1_model_path ${BEST_STG1_MODEL_PATH}} \
            ${BEST_STG2_MODEL_PATH:+--best_stg2_model_path ${BEST_STG2_MODEL_PATH}} \
            >> ${LOG_FILE} 2>&1 &

echo "Process started, PID: $!"
echo "Log file: ${LOG_FILE}"
echo "To monitor: tail -f ${LOG_FILE}"
echo "To stop: kill $!"

