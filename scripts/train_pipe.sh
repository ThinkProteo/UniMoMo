#!/bin/bash

SAVE_DIR=$1
AE_CONFIG=$2
LDM_CONFIG=$3

export CUDA_VISIBLE_DEVICES=${GPU}

echo "Training AutoEncoder"
python train.py --config ${AE_CONFIG} --trainer.config.save_dir=${SAVE_DIR}/AE_unimomo
if [ $? -eq 0 ]; then
    echo "Succeeded in training AutoEncoder"
else
    echo "Failed in training AutoEncoder"
    exit 1
fi

echo "Overwriting args"
# Find the latest version directory
LATEST_VERSION=$(ls -d ${SAVE_DIR}/AE_unimomo/version_* | sort -V | tail -n 1)
echo "Using checkpoint from: ${LATEST_VERSION}"

# Get the top checkpoint
TOPK_MAP="${LATEST_VERSION}/checkpoint/topk_map.txt"
if [ ! -f "${TOPK_MAP}" ]; then
    echo "Error: topk_map.txt not found at ${TOPK_MAP}"
    exit 1
fi

TOP_CKPT=$(cat ${TOPK_MAP} | head -n 1)
AE_CKPT_PATH="${LATEST_VERSION}/checkpoint/${TOP_CKPT}"

if [ ! -f "${AE_CKPT_PATH}" ]; then
    echo "Error: Checkpoint not found at ${AE_CKPT_PATH}"
    exit 1
fi

echo "Using AutoEncoder checkpoint: ${AE_CKPT_PATH}"

echo "Training LDM"
python train.py --config ${LDM_CONFIG} \
    --trainer.config.save_dir=${SAVE_DIR}/LDM_unimomo \
    --model.autoencoder_ckpt="${AE_CKPT_PATH}"

if [ $? -eq 0 ]; then
    echo "Succeeded in training LDM"
else
    echo "Failed in training LDM"
    exit 1
fi
