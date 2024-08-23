export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 

export PYTHONPATH='.'
MODEL_ROOT=/mnt/bn/bytenn-yg2/pretrained_models
# MODEL_ROOT=pretrained_models
# MODEL_ROOT=/mnt/bn/bytenn-lq2/pretrained_models
# RESULT_ROOT=/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results
# RESULT_ROOT=/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results
MODEL_LARGE=CompVis--stable-diffusion-v1-4
MODEL_SMALL=lcm_ours_tiny_sd14
TEACHER_MODEL=$MODEL_ROOT/$MODEL_LARGE

GPU_NUM=1








#Hybrid inference with LCM models
MODEL_LARGE=runwayml--stable-diffusion-v1-4
PATH_MODEL_LARGE=results/lcm_sd14_2w/checkpoint-20000
PATH_MODEL_SMALL=results/lcm_ours_tiny_sd14/checkpoint-20000
step_list=("0,8" "4,4" "6,2", "8,0")


for STEP in ${step_list[@]}
do
    OUTPUT_DIR=results/HybridSD_LCM_guidance7_paper_visual/$MODEL_LARGE-$MODEL_SMALL-$STEP
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid_LCM.py \
            --model_id $PATH_MODEL_LARGE $PATH_MODEL_SMALL \
            --steps $STEP  \
            --pretrained_teacher_model  $TEACHER_MODEL\
            --prompts_file examples/hybrid_sd/prompts_lcm2.txt \
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

