export PYTHONPATH='.'

MODEL_NAME="/mnt/bn/bytenn-yg2/pretrained_models/SG161222--Realistic_Vision_V5.1_noVAE"
# TRAIN_DATA_DIR="/mnt/bn/bytenn-yg2/datasets/laion_aes/preprocessed_11k"
TRAIN_DATA_DIR="/mnt/bn/bytenn-yg2/datasets/laion_aes/preprocessed_212k"

UNET_NAME="bk_small" # option: ["bk_base", "bk_small", "bk_tiny"]
GPU_NUM=0

BATCH_SIZE=16
GRAD_ACCUMULATION=4

# EXP_NAMES=("prune_oneshot/ratio-0.5")
EXP_NAMES=("prune_combined/a4_8" "prune_oneshot/ratio-0.75")

for exp_name in ${EXP_NAMES[@]};
do
  echo "exp_name = $exp_name"
  UNET_CONFIG_PATH="results/NaivePrune/bk-sdm-small/$exp_name"
  OUTPUT_DIR=$UNET_CONFIG_PATH # please adjust it if needed
  echo "output_dir = $OUTPUT_DIR"

  StartTime=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU_NUM accelerate launch publics/BK-SDM/src/kd_finetune_t2i.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --train_data_dir $TRAIN_DATA_DIR\
    --dataset_name laion_aes \
    --resolution 512 --center_crop --random_flip \
    --train_batch_size $BATCH_SIZE \
    --gradient_checkpointing \
    --mixed_precision="fp16" \
    --learning_rate 5e-05 \
    --max_grad_norm 1 \
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --report_to="all" \
    --max_train_steps=50000 \
    --seed 1234 \
    --checkpoints_total_limit 10 \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --checkpointing_steps 5000 \
    --valid_steps 5000 \
    --valid_prompt "A small white dog looking into a camera." "a brown and white cat staring off with pretty green eyes." \
    --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
    --unet_config_path $UNET_CONFIG_PATH --unet_config_name $UNET_NAME \
    --output_dir $OUTPUT_DIR

  EndTime=$(date +%s)
  echo "** KD training takes $(($EndTime - $StartTime)) seconds."
done

