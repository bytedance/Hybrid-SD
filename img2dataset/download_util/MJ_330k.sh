#!/bin/bash

#http proxy
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org

export PATH=$PATH:/mnt/bn/bytenn-data2/liuhj/pylib
# Index(['URL', 'TEXT', 'WIDTH', 'HEIGHT', 'similarity', 'punsafe', 'pwatermark', 'AESTHETIC_SCORE', 'hash'], dtype='object')
# img2dataset --url_list /mnt/bn/bytenn-data2/Laion_aesthetics_5plus_1024_33M --input_format "parquet"\
#          --url_col "URL" --caption_col "TEXT" --output_format webdataset\
#            --output_folder /mnt/bn/bytenn-data2/liuhj/test --processes_count 16 --thread_count 64 --image_size 1024\
#             --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
#              --save_additional_columns '["similarity","hash","WIDTH","HEIGHT","punsafe","pwatermark","AESTHETIC_SCORE"]' --enable_wandb True


img2dataset --url_list /mnt/bn/bytenn-data2/liuhj/MJ_dataset/meta_310k --input_format "parquet"\
        --url_col "image_url" --caption_col "prompt" --output_format files\
        --output_folder /mnt/bn/bytenn-data2/liuhj/MJ_dataset/310K --processes_count 1 --thread_count 16 --image_size 2048\
        --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True \
        --max_shard_retry 5 --incremental incremental \
        --enable_wandb False \
        #--distributor multiprocessing
