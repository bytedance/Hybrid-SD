export PYTHONPATH='.'



MODEL_DIR="pretrained_models/CompVis--stable-diffusion-v1-4"
TRAIN_DATA_DIR="datasets/laion2b_en_aesthetics/data"  
OUTPUT_DIR="results/lcm_sd14_2w_news" # please adjust it if needed

BATCH_SIZE=12
GRAD_ACCUMULATION=2

StartTime=$(date +%s)
GPU_NUM="4,5"

accelerate config default
CUDA_VISIBLE_DEVICES=$GPU_NUM accelerate launch \
  --main_process_port 23131 \
  --num_processes=2 \
  --multi_gpu \
  --num_machines=1 \
  examples/prune_sd/train_lcm_distill_sd_wds.py \
  --pretrained_teacher_model=$MODEL_DIR \
  --output_dir=$OUTPUT_DIR \
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



