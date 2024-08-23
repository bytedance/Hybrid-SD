# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 


# ### laion_aes datasets, including 11k, 212k, 2256k
# MODEL_NAME="runwayml/stable-diffusion-v1-5"
# UNET_CONFIG_PATH="./src/unet_config"
# UNET_NAME="bk_small" # option: ["bk_base", "bk_small", "bk_tiny"]
# # TRAIN_DATA_DIR="./data/laion_aes/preprocessed_2256k" # please adjust it if needed
# # TRAIN_DATA_DIR="./data/laion_aes/preprocessed_212k" # please adjust it if needed
# TRAIN_DATA_DIR="./data/laion_aes/preprocessed_11k" # please adjust it if needed
# OUTPUT_DIR="./results/laion_aes/kd_"$UNET_NAME"_2m" # please adjust it if needed

# BATCH_SIZE=64
# GRAD_ACCUMULATION=4

# StartTime=$(date +%s)

# accelerate launch \
#   --main_process_port 12348 \
#   --config_file multi_gpu.yaml \
#   --num_processes=3 \
#   --num_machines=1 \
#   --gpu_ids='5,6,7' \
#    src/kd_train_t2i.py \
#   --pretrained_model_name_or_path $MODEL_NAME \
#   --use_ema \
#   --seed 1234 \
#   --resolution 512 --center_crop --random_flip \
#   --gradient_checkpointing \
#   --train_data_dir $TRAIN_DATA_DIR\
#   --train_batch_size $BATCH_SIZE \
#   --mixed_precision="fp16" \
#   --learning_rate 5e-05 \
#   --max_grad_norm 1 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps=0 \
#   --max_train_steps=400000 \
#   --gradient_accumulation_steps $GRAD_ACCUMULATION \
#   --checkpointing_steps 5000 \
#   --checkpoints_total_limit 10 \
#   --valid_steps 1000 \
#   --enable_xformers_memory_efficient_attention \
#   --use_copy_weight_from_teacher \
#   --lambda_sd 1.0 \
#   --lambda_kd_output 1.0 \
#   --lambda_kd_feat 1.0 \
#   --report_to="wandb" \
#   --unet_config_path $UNET_CONFIG_PATH \
#   --unet_config_name $UNET_NAME \
#   --output_dir $OUTPUT_DIR

# EndTime=$(date +%s)
# echo "** KD training takes $(($EndTime - $StartTime)) seconds."



# ### laion-art datasets
# MODEL_NAME="runwayml/stable-diffusion-v1-5"
# UNET_CONFIG_PATH="./src/unet_config"
# UNET_NAME="bk_small" # option: ["bk_base", "bk_small", "bk_tiny"]
# TRAIN_DATA_DIR="/mnt/bn/bytenn-data2/laion-art" # please adjust it if needed
# OUTPUT_DIR="/mnt/bn/bytenn-data2/sd_models/laion_art/kd_"$UNET_NAME"_2m" # please adjust it if needed

# BATCH_SIZE=1
# GRAD_ACCUMULATION=4

# StartTime=$(date +%s)

# accelerate launch \
#   --main_process_port 12348 \
#   --config_file single_gpu.yaml \
#   --num_processes=1 \
#   --num_machines=1 \
#    src/kd_train_t2i.py \
#   --pretrained_model_name_or_path $MODEL_NAME \
#   --use_ema \
#   --seed 1234 \
#   --resolution 512 --center_crop --random_flip \
#   --gradient_checkpointing \
#   --dataset_name $TRAIN_DATA_DIR\
#   --train_batch_size $BATCH_SIZE \
#   --mixed_precision="fp16" \
#   --learning_rate 1e-05 \
#   --max_grad_norm 1 \
#   --lr_scheduler="constant" \
#   --lr_warmup_steps 0 \
#   --max_train_steps 100 \
#   --gradient_accumulation_steps $GRAD_ACCUMULATION \
#   --checkpointing_steps 10 \
#   --checkpoints_total_limit 10 \
#   --valid_steps 10 \
#   --enable_xformers_memory_efficient_attention \
#   --use_copy_weight_from_teacher \
#   --lambda_sd 1.0 \
#   --lambda_kd_output 0.5 \
#   --lambda_kd_feat 0.5 \
#   --report_to="wandb" \
#   --unet_config_path $UNET_CONFIG_PATH \
#   --unet_config_name $UNET_NAME \
#   --output_dir $OUTPUT_DIR

# EndTime=$(date +%s)
# echo "** KD training takes $(($EndTime - $StartTime)) seconds."



### Implement layer replacement in a more elegant way ###
### TODO: bugfix for ema training
### laion-art datasets
MODEL_NAME="runwayml/stable-diffusion-v1-5"
UNET_CONFIG_PATH="./src/unet_config"
UNET_NAME="bk_small" # option: ["bk_small", "bk_tiny"]
TRAIN_DATA_DIR="/mnt/bn/bytenn-data2/laion-art" # please adjust it if needed
OUTPUT_DIR="/mnt/bn/bytenn-data2/sd_models/laion_art/kd_"$UNET_NAME"_2m" # please adjust it if needed

BATCH_SIZE=1
GRAD_ACCUMULATION=4

StartTime=$(date +%s)

accelerate launch \
  --main_process_port 12348 \
  --config_file multi_gpu.yaml \
  --num_processes=4 \
  --num_machines=1 \
   src/kd_train_t2i.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --seed 1234 \
  --resolution 512 --center_crop --random_flip \
  --gradient_checkpointing \
  --dataset_name $TRAIN_DATA_DIR\
  --train_batch_size $BATCH_SIZE \
  --mixed_precision="fp16" \
  --learning_rate 1e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps 0 \
  --max_train_steps 100 \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --checkpointing_steps 10 \
  --checkpoints_total_limit 10 \
  --valid_steps 10 \
  --enable_xformers_memory_efficient_attention \
  --use_copy_weight_from_teacher \
  --lambda_sd 1.0 \
  --lambda_kd_output 0.5 \
  --lambda_kd_feat 0.5 \
  --report_to="wandb" \
  --unet_config_path $UNET_CONFIG_PATH \
  --unet_config_name $UNET_NAME \
  --output_dir $OUTPUT_DIR

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."



