MODELPATH="outputs/taesd_pretrained"
INPUT="outputs/train_tinyvae_ldm/coco2017/coco2017_9999"



export CUDA_VISIBLE_DEVICES=0,1


python3 evaluation/evaluation.py --pretrained_model_name_or_path $MODELPATH \
                    --if_baseline False \
                    --model_name "tinyvae" \
                    --device_ids "0,1" \
                    --ckpt_name "checkpoint-repro/vae.bin" \
                    --input_dir $INPUT \
                    --input_root_gen $INPUT \
                    --if_fp16 False  

