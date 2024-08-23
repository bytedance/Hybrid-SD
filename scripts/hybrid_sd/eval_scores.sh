# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Please ensure that the following libraries are successfully installed:
#   for IS, https://github.com/toshas/torch-fidelity
#   for FID, https://github.com/mseitzer/pytorch-fid
#   for CLIP score, https://github.com/mlfoundations/open_clip
# ------------------------------------------------------------------------------------

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 

export PYTHONPATH='.'

GPU_NUM=1

# model_list=("small-sd_25" "tiny-sd_25" "stable-diffusion-v1-5_25" "bk-sdm-base-2m_25" "bk-sdm-base_25" "bk-sdm-small-2m_25" "bk-sdm-small_25" "bk-sdm-tiny-2m_25" "bk-sdm-tiny_25")
# let model_list_length=${#model_list[@]}-1

# model_list=("stable-diffusion-v1-4bk-sdm-small-2mbk-sdm-tiny-2m20_5_0" "stable-diffusion-v1-4bk-sdm-small-2mbk-sdm-tiny-2m20_0_5" "stable-diffusion-v1-4bk-sdm-small-2mbk-sdm-tiny-2m15_10_0" "stable-diffusion-v1-4bk-sdm-small-2mbk-sdm-tiny-2m15_5_5" "stable-diffusion-v1-4bk-sdm-small-2mbk-sdm-tiny-2m15_0_10" "stable-diffusion-v1-4bk-sdm-small-2mbk-sdm-tiny-2m10_15_0" "stable-diffusion-v1-4bk-sdm-small-2mbk-sdm-tiny-2m10_0_15")
# let model_list_length=${#model_list[@]}-1

# model_list=("Realistic_Vision_V4.0small-sdtiny-sd20_5_0" "Realistic_Vision_V4.0small-sdtiny-sd20_0_5" "Realistic_Vision_V4.0small-sdtiny-sd15_10_0" "Realistic_Vision_V4.0small-sdtiny-sd15_5_5" "Realistic_Vision_V4.0small-sdtiny-sd15_0_10" "Realistic_Vision_V4.0small-sdtiny-sd10_15_0" "Realistic_Vision_V4.0small-sdtiny-sd10_10_5" "Realistic_Vision_V4.0small-sdtiny-sd10_5_10" "Realistic_Vision_V4.0small-sdtiny-sd10_0_15" "Realistic_Vision_V4.0small-sdtiny-sd5_20_0" "Realistic_Vision_V4.0small-sdtiny-sd5_10_10" "Realistic_Vision_V4.0small-sdtiny-sd5_0_20" "Realistic_Vision_V4.0small-sdtiny-sd2_23_0" "Realistic_Vision_V4.0small-sdtiny-sd2_0_23")
# let model_list_length=${#model_list[@]}-1

model_list=("bk-sdm-small_25")
let model_list_length=${#model_list[@]}-1

for index in $(seq 0 ${model_list_length})
do
    MODEL_ID=${model_list[index]}
    IMG_PATH=./results/$MODEL_ID/im256

    echo "=== Inception Score (IS) ==="
    IS_TXT=./results/$MODEL_ID/im256_is.txt
    fidelity --gpu $GPU_NUM --isc --input1 $IMG_PATH | tee $IS_TXT
    echo "============"

    echo "=== Fréchet Inception Distance (FID) ==="
    FID_TXT=./results/$MODEL_ID/im256_fid.txt
    NPZ_NAME_gen=./results/$MODEL_ID/im256_fid.npz
    rm -rf $NPZ_NAME_gen
    NPZ_NAME_real=/mnt/bd/bytenn-lq/datasets/mscoco_val2014_41k_full/real_im256.npz
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid --save-stats $IMG_PATH $NPZ_NAME_gen
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 -m pytorch_fid $NPZ_NAME_real $NPZ_NAME_gen | tee $FID_TXT
    rm -rf $NPZ_NAME_gen
    echo "============"

done

for index in $(seq 0 ${model_list_length})
do
    MODEL_ID=${model_list[index]}
    IMG_PATH=./results/$MODEL_ID/im256
    echo "=== CLIP Score ==="
    CLIP_TXT=./results/$MODEL_ID/im256_clip.txt
    DATA_LIST=/mnt/bd/bytenn-lq/datasets//mscoco_val2014_30k/metadata.csv
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 evaluation/clip_score.py --img_dir $IMG_PATH --data_list $DATA_LIST --save_txt $CLIP_TXT
    echo "============"
done