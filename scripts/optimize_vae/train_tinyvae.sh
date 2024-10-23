NUM_GPUS=2
BATCH_SIZE=24
ACC_STEPS=2

mkdir -p publics
cd publics
git clone https://github.com/CompVis/taming-transformers.git
cd ..
pip install -e publics/taming-transformers

CUDA_VISIBLE_DEVICES=6,7 accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port 23333 \
    examples/optimize_vae/train_tinyvae.py \
    --device_ids '6, 7' \
    --output_dir 'outputs_ldm/train_tinyvae' \
    --learning_rate 1e-5 \
    --lr_scheduler "cosine" \
    --lr_warmup_steps 0 \
    --adam_weight_decay 0.01 \
    --seed 42 \
    --gradient_accumulation_steps $ACC_STEPS \
    --train_batch_size 24 \
    --num_train_epochs 50000 \
    --max_train_steps 100000 \
    --log_steps 100 \
    --checkpointing_steps 1000 \
    --resolution 512 \
    --mixed_precision no \
    --train_data_dir datasets/Laion_aesthetics_5plus_1024_33M/Laion33m_data_test \
    --pretrained_model_name_or_path "pretrained_models/runwayml--stable-diffusion-v1-5"  \
    --student_model_name_or_path  "pretrained_models/madebyollin--taesd"  \
    --experiment_name fintune_dino_combine_pixelfilter \
    --disc_start 5000 \
    --add_lq_input True \
    --visual_path datasets/taesd_visual \
    --real_path datasets/coco2017_resize \
    --report_to wandb


