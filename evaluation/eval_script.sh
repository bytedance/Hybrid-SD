MODELPATH="/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/taesd_pretrained"
INPUT="/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/finetune_tinyvae_ldm/coco2017/coco2017_9999"


#export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
export CUDA_VISIBLE_DEVICES=0,1


python3 evaluation/evaluation_lhj.py --pretrained_model_name_or_path $MODELPATH \
                    --if_baseline False \
                    --model_name "tinyvae" \
                    --device_ids "0,1" \
                    --ckpt_name "checkpoint-repro/vae.bin" \
                    --input_dir $INPUT \
                    --input_root_gen $INPUT \
                    --if_fp16 False  


#/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/distill_tiny_vae_straight/checkpoint-56000
