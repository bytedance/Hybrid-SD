export PYTHONPATH='.'
MODEL_ROOT=pretrained_models
MODEL_LARGE=CompVis--stable-diffusion-v1-4
MODEL_SMALL=nota-ai--bk-sdm-tiny
GPU_NUM=1

# Hybrid inference with LCM models
MODEL_LARGE=CompVis--stable-diffusion-v1-4 
TEACHER_MODEL_PATH=$MODEL_ROOT/$MODEL_LARGE
PATH_MODEL_LARGE="results/lcm_sd14_2w/checkpoint-20000"
PATH_MODEL_SMALL="results/lcm_ours_tiny_sd14/checkpoint-20000"
step_list=("0,8" "4,4"  "8,0")


for STEP in ${step_list[@]}
do
    OUTPUT_DIR=results/HybridSD_LCM_guidance7/$MODEL_LARGE-$MODEL_SMALL-$STEP
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid_LCM.py \
            --model_id $PATH_MODEL_LARGE $PATH_MODEL_SMALL\
            --pretrained_teacher_model $TEACHER_MODEL_PATH\
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



