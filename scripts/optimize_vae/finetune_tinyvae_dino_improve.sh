OUTDIR="/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/finetune_tinyvae_dino_improveW2"
export WANDB_DIR=$OUTDIR
# training from scratch

export CUDA_VISIBLE_DEVICES=6,7
# tmux 1
NUM_GPUS=2

BATCH_SIZE=24
ACC_STEPS=2

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org


pip install prefetch_generator PyWavelets
pip install -U byted-wandb -i https://bytedpypi.byted.org/simple
pip install -e /mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/publics/taming-transformers
pip install -e /mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/publics/pytorch_wavelets


accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port 21225\
    /mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/examples/optimize_vae/finetune_tinyvae_dino_improve.py \
    --output_dir $OUTDIR \
    --learning_rate 1e-5 \
    --lr_scheduler "cosine" \
    --lr_warmup_steps 0 \
    --adam_weight_decay 0.01 \
    --seed 42 \
    --gradient_accumulation_steps $ACC_STEPS \
    --train_batch_size $BATCH_SIZE \
    --num_train_epochs 50000 \
    --max_train_steps 100000 \
    --log_steps 100 \
    --checkpointing_steps 2000\
    --resolution 512 \
    --mixed_precision no \
    --train_data_dir /mnt/bn/bytenn-yg2/datasets/Laion_aesthetics_5plus_1024_33M/Laion33m_data_test \
    --pretrained_model_name_or_path "/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5"  \
    --student_model_name_or_path  "/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd"  \
    --disc_start 5000 --report_to wandb
