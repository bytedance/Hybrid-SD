
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 

export PYTHONPATH='.'

# DATA_ROOT=/mnt/bn/bytenn-lq2/datasets
DATA_ROOT=./datasets
MODEL_ROOT=./pretrained_models
# MODEL_LARGE=SG161222--Realistic_Vision_V5.1_noVAE
# MODEL_SMALL=nota-ai--bk-sdm-small
MODEL_LARGE=CompVis--stable-diffusion-v1-4
MODEL_SMALL=nota-ai--bk-sdm-tiny

PATH_MODEL_LARGE=$MODEL_ROOT/$MODEL_LARGE
PATH_MODEL_SMALL=$MODEL_ROOT/$MODEL_SMALL
# PATH_MODEL_LARGE="CompVis/stable-diffusion-v1-4"
# PATH_MODEL_SMALL="nota-ai/bk-sdm-tiny"


GPU_NUM=1
BATCH_SIZE=128

generate() {
    echo "PATH_MODEL_LARGE=$1"
    echo "PATH_MODEL_SMALL=$2"
    echo "OUTPUT_DIR=$3"
    echo "STPES=$4"
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid_batch.py \
            --model_id $1 $2 \
            --output_dir $3 \
            --steps $4  \
            --data_list $DATA_ROOT/mscoco_val2014_30k/metadata.csv \
            --seed 1674753452 \
            --img_sz 512 \
            --img_resz 256 \
            --batch_sz $BATCH_SIZE \
            --num_images 1 \
            --num_images_per_prompt 1 \
            --enable_xformers_memory_efficient_attention \
            --use_dpm_solver \
            --guidance_scale 7
}

calc_is() {
    echo "=== Inception Score (IS) ==="
    IS_TXT=$1/im256_is.txt
    fidelity --gpu $GPU_NUM --isc --input1 $1/img256 | tee $IS_TXT
    echo "============"
}

calc_fid(){
    echo "=== Fréchet Inception Distance (FID) ==="
    FID_TXT=$1/im256_fid.txt
    NPZ_NAME_gen=$1/im256_fid.npz
    rm -rf $NPZ_NAME_gen
    NPZ_NAME_real=$DATA_ROOT/mscoco_val2014_41k_full/real_im256.npz
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $1/im256 $NPZ_NAME_gen --batch-size $BATCH_SIZE
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_real $NPZ_NAME_gen --batch-size $BATCH_SIZE | tee $FID_TXT
    rm -rf $NPZ_NAME_gen
    echo "============"
}

calc_clip() {
    echo "=== CLIP Score ==="
    CLIP_TXT=$1/im256_clip.txt
    DATA_LIST=$DATA_ROOT/mscoco_val2014_30k/metadata.csv
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 evaluation/clip_score.py --img_dir $1/im256 --data_list $DATA_LIST --save_txt $CLIP_TXT
    echo "============"
}

STEP_LIST=("0,25" "5,20" "10,15" "15,10" "20,5" "25,0")

for STEP in ${STEP_LIST[@]};
do
    export OUTPUT_DIR=results/HybridSD_pndm/$MODEL_LARGE-$MODEL_SMALL-$STEP

    generate $PATH_MODEL_LARGE $PATH_MODEL_SMALL $OUTPUT_DIR $STEP
    calc_is $OUTPUT_DIR
    calc_fid $OUTPUT_DIR
    # calc_clip $OUTPUT_DIR
done

