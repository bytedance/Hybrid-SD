export PYTHONPATH='.'
MODEL_ROOT=pretrained_models
MODEL_LARGE=stabilityai--stable-diffusion-xl-base-1.0
MODEL_SMALL=koala-700m

PATH_MODEL_LARGE=$MODEL_ROOT/$MODEL_LARGE
PATH_MODEL_SMALL=pretrained_models/koala-700m
GPU_NUM=1





# Hybrid inference with SDXL models
step_list=("25,0" "10,15" "15,10" "0,25") 


for STEP in ${step_list[@]}
do
    OUTPUT_DIR=results/HybridSDXL_dpm_guidance7_sdxl_prompts_cfg_hybrid/$MODEL_LARGE-$MODEL_SMALL-$STEP
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid_sdxl.py \
            --model_id $PATH_MODEL_LARGE $PATH_MODEL_SMALL\
            --steps $STEP  \
            --prompts_file examples/hybrid_sd/prompts_sdxl_hybrid.txt \
            --seed 1234 \
            --img_sz 1024 \
            --output_dir $OUTPUT_DIR \
            --num_images_per_prompt 1 \
            --num_images 1 \
            --enable_xformers_memory_efficient_attention \
            --save_middle \
            --use_dpm_solver \
            --weight_dtype fp32 \
            --guidance_scale 7
done

