export PYTHONPATH='.'

MODEL_ID=/mnt/bn/bytenn-lq2/pretrained_models/SG161222--Realistic_Vision_V5.1_noVAE
OUTPUT_DIR=results/SG161222--Realistic_Vision_V5.1_noVAE
IMG_PATH=$OUTPUT_DIR/im256
GPU_NUM=0
FID_BATCH_SIZE=100
DATA_ROOT=/mnt/bn/bytenn-lq2/datasets

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/generate_batch.py \
        --model_id $MODEL_ID \
        --steps 25  \
        --data_list $DATA_ROOT/mscoco_val2014_30k/metadata.csv \
        --seed 1674753452 \
        --img_sz 512 \
        --img_resz 256 \
        --batch_sz 128 \
        --num_images 1 \
        --num_images_per_prompt 1 \
        --output_dir $OUTPUT_DIR \
        --enable_xformers_memory_efficient_attention 

echo "=== Inception Score (IS) ==="
IS_TXT=$OUTPUT_DIR/im256_is.txt
fidelity --gpu $GPU_NUM --isc --input1 $IMG_PATH | tee $IS_TXT
echo "============"

echo "=== Fréchet Inception Distance (FID) ==="
FID_TXT=$OUTPUT_DIR/im256_fid.txt
NPZ_NAME_gen=$OUTPUT_DIR/im256_fid.npz
rm -rf $NPZ_NAME_gen
NPZ_NAME_real=$DATA_ROOT/mscoco_val2014_41k_full/real_im256.npz
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $IMG_PATH $NPZ_NAME_gen --batch-size $FID_BATCH_SIZE
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_real $NPZ_NAME_gen --batch-size $FID_BATCH_SIZE | tee $FID_TXT
rm -rf $NPZ_NAME_gen
echo "============"

echo "=== CLIP Score ==="
CLIP_TXT=$OUTPUT_DIR/im256_clip.txt
DATA_LIST=/mnt/bn/bytenn-lq2/datasets/mscoco_val2014_30k/metadata.csv
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 evaluation/clip_score.py --img_dir $IMG_PATH --data_list $DATA_LIST --save_txt $CLIP_TXT
echo "============"