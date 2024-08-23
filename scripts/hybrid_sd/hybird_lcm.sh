export PYTHONPATH='.'
MODEL_ROOT=/mnt/bn/bytenn-yg2/pretrained_models
MODEL_LARGE=CompVis--stable-diffusion-v1-4
MODEL_SMALL=nota-ai--bk-sdm-tiny
GPU_NUM=1

# Hybrid inference with LCM models
MODEL_LARGE=runwayml--stable-diffusion-v1-4
PATH_MODEL_LARGE="/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/results/lcm_sd14_2w/checkpoint-20000"
PATH_MODEL_SMALL=/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/results/nota-ai--bk-sdm-tiny_LCM/checkpoint-20000
step_list=("0,8" "4,4"  "8,0")


for STEP in ${step_list[@]}
do
    OUTPUT_DIR=results/HybridSD_LCM_guidance7/$MODEL_LARGE-$MODEL_SMALL-$STEP
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid_LCM.py \
            --model_id $PATH_MODEL_LARGE $PATH_MODEL_SMALL\
            --steps $STEP  \
            --prompts_file examples/hybrid_sd/prompts_realistic.txt \
            --seed 1674753452 \
            --img_sz 512 \
            --output_dir $OUTPUT_DIR \
            --num_images_per_prompt 1 \
            --num_images 1 \
            --enable_xformers_memory_efficient_attention \
            --save_middle \
            --use_dpm_solver \
            --guidance_scale 7
done



