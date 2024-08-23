export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 
export PYTHONPATH='.'


### laion-art datasets
### CompVis/stable-diffusion-v1-4 nota-ai/bk-sdm-small-2m nota-ai/bk-sdm-tiny-2m
### SG161222/Realistic_Vision_V4.0 segmind/small-sd segmind/tiny-sd
# MODEL_NAME="SG161222/Realistic_Vision_V5.1_noVAE"
MODEL_NAME="nota-ai/bk-sdm-small"
# TRAIN_DATA_DIR="/mnt/bn/bytenn-data2/laion-art" # please adjust it if needed
TRAIN_DATA_DIR='datasets/preprocessed_11k'
# OUTPUT_DIR="results/toy_lcm_Realistic_Vision_V5.1_noVAE" # please adjust it if needed
OUTPUT_DIR="results/toy_lcm_small" # please adjust it if needed

BATCH_SIZE=16
GRAD_ACCUMULATION=4

StartTime=$(date +%s)
GPU_NUM=2

CUDA_VISIBLE_DEVICES=$GPU_NUM accelerate launch \
  --main_process_port 23331 \
  --num_processes=4 \
  --num_machines=1 \
   publics/BK-SDM/src/lcm_train.py \
  --pretrained_teacher_model $MODEL_NAME \
  --unet_config_path="publics/BK-SDM/src/unet_config" \
  --unet_config_name="bk_small" \
  --output_dir $OUTPUT_DIR \
  --resolution 512 --center_crop --random_flip \
  --seed 1234 \
  --gradient_checkpointing \
  --enable_xformers_memory_efficient_attention \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --learning_rate=1e-6 \
  --loss_type="huber" \
  --ema_decay=0.95 \
  --adam_weight_decay=0.0 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps 0 \
  --max_train_steps 50000 \
  --max_train_samples 2000 \
  --dataloader_num_workers=8 \
  --validation_steps=200 \
  --checkpointing_steps=200 \
  --checkpoints_total_limit=10 \
  --mixed_precision="fp16" \
  --dataset_name="laion-aes" \
  --train_data_dir $TRAIN_DATA_DIR \
  --report_to="wandb" \
  --resume_from_checkpoint latest \
  --valid_prompt "a golden vase with different flowers." "a brown and white cat staring off with pretty green eyes." 


EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."



