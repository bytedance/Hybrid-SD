export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 
export PYTHONPATH='.'


### laion-art datasets
### CompVis/stable-diffusion-v1-4 nota-ai/bk-sdm-small-2m nota-ai/bk-sdm-tiny-2m
### SG161222/Realistic_Vision_V4.0 segmind/small-sd segmind/tiny-sd
# MODEL_NAME="SG161222/Realistic_Vision_V5.1_noVAE"
#MODEL_NAME="nota-ai/bk-sdm-small"
# TRAIN_DATA_DIR="/mnt/bn/bytenn-data2/laion-art" # please adjust it if needed
# MODEL_DIR="/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5"
MODEL_DIR="/mnt/bn/bytenn-yg2/pretrained_models/CompVis--stable-diffusion-v1-4"
#MODEL_DIR="/mnt/bn/bytenn-yg2/pretrained_models/nota-ai--bk-sdm-tiny"
TRAIN_DATA_DIR="/mnt/bn/bytenn-yg2/datasets/laion2b_en_aesthetics/data"

#STUDENT_DIR="/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/results/NaivePrune/bk-sdm-tiny/prune_combined_v2/a10_b20"
#STUDENT_DIR="results/finetune/NaivePrune/a19_b21/unet_finetuned/2024-08-01-20-43-15"
#STUDENT_DIR="/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results/NaivePrune/bk-sdm-tiny/a19_b21/checkpoint-50000"
STUDENT_DIR="/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results/NaivePrune_SD14/bk-sdm-tiny/prune_combined_v2/a19_b21/2024-05-15-22-00-19/checkpoint-50000"
#STUDENT_DIR="/mnt/bn/bytenn-yg2/pretrained_models/nota-ai--bk-sdm-tiny"
#'/mnt/bn/bytenn-yg2/datasets/laion2b_en_aesthetics/data'        #'/mnt/bn/bytenn-yg2/datasets/laion_aes/preprocessed_11k' \
#OUTPUT_DIR="results/ours_tiny/lcm_a19_b21_tiny_teacher" # please adjust it if needed

OUTPUT_DIR="results/lcm_sd14_ours_224_tiny_teacher"
BATCH_SIZE=12
GRAD_ACCUMULATION=2

StartTime=$(date +%s)
GPU_NUM="2,3"

accelerate config default
CUDA_VISIBLE_DEVICES=$GPU_NUM accelerate launch \
  --main_process_port 23105 \
  --num_processes=2 \
  --multi_gpu \
  --num_machines=1 \
  examples/prune_sd/train_lcm_distill_bk_sdm_wds_tiny_teacher.py \
  --pretrained_teacher_model=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
  --tiny_student_model=$STUDENT_DIR \
  --mixed_precision=fp16 \
  --resolution=512 \
  --learning_rate=1e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
  --max_train_steps=20000 \
  --max_train_samples=4000000 \
  --dataloader_num_workers=8 \
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



