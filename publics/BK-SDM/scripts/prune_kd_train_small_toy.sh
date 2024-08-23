# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------
export PYTHONPATH='.'

MODEL_NAME="/mnt/bn/ycq-lq/hf_models/nota-ai--bk-sdm-small"
# TRAIN_DATA_DIR="/mnt/bn/bytenn-yg2/datasets/laion_aes/preprocessed_11k" # please adjust it if needed
TRAIN_DATA_DIR="/mnt/bn/ycq-lq/data/laion_aes/preprocessed_11k" # please adjust it if needed
UNET_CONFIG_PATH="./src/unet_config"

UNET_NAME="bk_small" # option: ["bk_base", "bk_small", "bk_tiny"]
OUTPUT_DIR="./results/toy_prune_kd_"$UNET_NAME # please adjust it if needed

BATCH_SIZE=2
GRAD_ACCUMULATION=4

StartTime=$(date +%s)

CUDA_VISIBLE_DEVICES=0 accelerate launch publics/BK-SDM/src/kd_train_t2i_prune_iteratively.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --dataset_name laion_aes \
  --train_data_dir $TRAIN_DATA_DIR \
  --resolution 512 --center_crop --random_flip \
  --train_batch_size $BATCH_SIZE \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate 5e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="all" \
  --max_train_steps=400000 \
  --seed 1234 \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --checkpointing_steps 2000 \
  --valid_steps 500 \
  --valid_prompt "a golden vase with different flowers." "a brown and white cat staring off with pretty green eyes." \
  --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
  --output_dir $OUTPUT_DIR 


EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."

