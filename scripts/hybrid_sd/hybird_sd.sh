export PYTHONPATH='.'
MODEL_ROOT=pretrained_models
MODEL_LARGE=CompVis--stable-diffusion-v1-4
MODEL_SMALL=hybrid-sd-224m

PATH_MODEL_LARGE=$MODEL_ROOT/$MODEL_LARGE
PATH_MODEL_SMALL=$MODEL_ROOT/cqyan/hybrid-sd-224m # path to our tiny model

GPU_NUM=0

step_list=("0,25" "10,15" "25,0")


for STEP in ${step_list[@]}
do
    OUTPUT_DIR=results/HybridSD_dpm_guidance7_visual/$MODEL_LARGE-$MODEL_SMALL-$STEP
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid.py \
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