export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 
export PYTHONPATH='.'


### laion-art datasets
### CompVis/stable-diffusion-v1-4 nota-ai/bk-sdm-small-2m nota-ai/bk-sdm-tiny-2m
### SG161222/Realistic_Vision_V4.0 segmind/small-sd segmind/tiny-sd
# MODEL_NAME="SG161222/Realistic_Vision_V5.1_noVAE"
#MODEL_NAME="nota-ai/bk-sdm-small"
# TRAIN_DATA_DIR="/mnt/bn/bytenn-data2/laion-art" # please adjust it if needed
MODEL_DIR="/mnt/bn/bytenn-yg2/pretrained_models/stabilityai--stable-diffusion-xl-base-1.0"
VAE_DIR="/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--sdxl-vae-fp16-fix"
TRAIN_DATA_DIR="/mnt/bn/bytenn-yg2/datasets/laion2b_en_aesthetics/data"
#'/mnt/bn/bytenn-yg2/datasets/laion2b_en_aesthetics/data'        #'/mnt/bn/bytenn-yg2/datasets/laion_aes/preprocessed_11k' \
OUTPUT_DIR="results/lcm_sdxl_2w" # please adjust it if needed


BATCH_SIZE=4
GRAD_ACCUMULATION=1

StartTime=$(date +%s)
GPU_NUM="0,1"

accelerate config default
CUDA_VISIBLE_DEVICES=$GPU_NUM accelerate launch \
  --main_process_port 23110 \
  --num_processes=2 \
  --multi_gpu \
  --num_machines=1 \
  examples/prune_sd/train_lcm_distill_sdxl_wds.py \
  --pretrained_teacher_model=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --pretrained_vae_model_name_or_path=$VAE_DIR \
  --mixed_precision=fp16 \
  --resolution=1024 \
  --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
  --max_train_steps=20000 \
  --max_train_samples=4000000 \
  --dataloader_num_workers=2\
  --use_8bit_adam \
  --train_shards_path_or_url=$TRAIN_DATA_DIR \
  --validation_steps=200 \
  --checkpointing_steps=1000 --checkpoints_total_limit=10 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_checkpointing --enable_xformers_memory_efficient_attention \
  --gradient_accumulation_steps=$GRAD_ACCUMULATION \
  --resume_from_checkpoint=latest \
  --report_to=wandb \
  --seed=1234

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."



