# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Please ensure that the following libraries are successfully installed:
#   for IS, https://github.com/toshas/torch-fidelity
#   for FID, https://github.com/mseitzer/pytorch-fid
#   for CLIP score, https://github.com/mlfoundations/open_clip
# ------------------------------------------------------------------------------------


export PYTHONPATH='.'

GPU_NUM=0

model_list=("reproduce/nota-ai--bk-sdm-small-diffusers-0.19.0-v2") 

let model_list_length=${#model_list[@]}-1



for index in $(seq 0 ${model_list_length})
do
    MODEL_ID=${model_list[index]}
    IMG_PATH=./results/$MODEL_ID/im256
    echo "=== CLIP Score ==="
    CLIP_TXT=./results/$MODEL_ID/im256_clip.txt
    DATA_LIST=datasets/mscoco_val2014_30k/metadata.csv
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 evaluation/clip_score.py --img_dir $IMG_PATH --data_list $DATA_LIST --save_txt $CLIP_TXT
    echo "============"
done