export PYTHONPATH='.'

CUDA_VISIBLE_DEVICES=0 python3 examples/prune_sd/generate_batch.py \
        --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--bk-sdm-small \
        --unet_path /mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/publics/BK-SDM/results/bs64_kd_bk_small/checkpoint-50000 \
        --steps 25  \
        --data_list /mnt/bn/bytenn-yg2/datasets/mscoco_val2014_30k/metadata.csv \
        --seed 1674753452 \
        --img_sz 512 \
        --img_resz 256 \
        --batch_sz 96 \
        --num_images 1 \
        --num_images_per_prompt 1 \
        --output_dir results/debug \
        --enable_xformers_memory_efficient_attention \
        --max_n_files 10


