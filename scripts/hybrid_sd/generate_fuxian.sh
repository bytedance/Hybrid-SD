
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 

export PYTHONPATH='.'

# SG161222/Realistic_Vision_V4.0 segmind/small-sd segmind/tiny-sd
# runwayml/stable-diffusion-v1-5 nota-ai/bk-sdm-base-2m nota-ai/bk-sdm-small-2m nota-ai/bk-sdm-tiny-2m
# CompVis/stable-diffusion-v1-4 nota-ai/bk-sdm-base    nota-ai/bk-sdm-small    nota-ai/bk-sdm-tiny


# python3 examples/hybrid_sd/generate.py \
#         --model_id segmind/tiny-sd \
#         --steps 25  \
#         --prompts_file examples/hybrid_sd/prompts.txt \
#         --seed 1674753452 \
#         --img_sz 512 \
#         --guidance_scale 7 \
#         --num_images_per_prompt 1 \
#         --output_dir results \
#         --num_images 1 \
#         --use_dpm_solver \
#         --enable_xformers_memory_efficient_attention \
#         --save_middle


model_list=("/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results/NaivePrune_SD14/bk-sdm-tiny/prune_combined_v2/a19_b21/2024-05-15-22-00-19")
let model_length=${#model_list[@]}-1
# --unet_path /mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/results/NaivePrune_SD14/bk-sdm-tiny/prune_combined_v2/a19_b21/2024-05-15-22-00-19/checkpoint-50000 \
for index in $(seq 0 ${model_length})
do
        model_id=${model_list[index]}
        CUDA_VISIBLE_DEVICES=0 python3 examples/hybrid_sd/generate_batch.py \
                --model_id ${model_id} \
                --steps 25  \
                --data_list /mnt/bn/bytenn-yg2/datasets/mscoco_val2014_30k/metadata.csv \
                --seed 1674753452 \
                --img_sz 512 \
                --img_resz 256 \
                --batch_sz 128 \
                --num_images 1 \
                --guidance_scale 7 \
                --num_images_per_prompt 1 \
                --output_dir results/HybridSD_dpm_guidance7_fuxian_generate_bs128_img128 \
                --use_dpm_solver \
                --enable_xformers_memory_efficient_attention \
                --save_middle
done