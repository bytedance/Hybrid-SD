export PYTHONPATH='.'

MODEL_ROOT="pretrained_models"
DATA_ROOT="datasets"

TRAIN_DATA_DIR="$DATA_ROOT/laion_aes/preprocessed_11k"  

MODEL_NAME=$MODEL_ROOT/CompVis--stable-diffusion-v1-4 
UNET_NAME="bk_tiny" # option: ["bk_base", "bk_small", "bk_tiny"]

GPU_NUM=1

BATCH_SIZE=16
GRAD_ACCUMULATION=4

EXP_NAMES=("a19_b21/unet")

for exp_name in ${EXP_NAMES[@]};
do
  echo "exp_name = $exp_name"
  UNET_CONFIG_PATH="results/NaivePrune/bk-sdm-tiny/prune_oneshot/$exp_name"
  OUTPUT_DIR="results/finetune/NaivePrune/${exp_name}_finetuned" # please adjust it if needed

  echo "output_dir = $OUTPUT_DIR"

  StartTime=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU_NUM accelerate launch  --num_processes ${GPU_NUM} --main_process_port 21101 \
  examples/prune_sd/kd_finetune_t2i.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --train_data_dir $TRAIN_DATA_DIR \
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
    --checkpoints_total_limit 5 \
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

