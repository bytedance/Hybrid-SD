#!/bin/bash

#http proxy
#export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org

# for i in {00000..00127}; do wget https://huggingface.co/datasets/kakaobrain/coyo-700m/resolve/main/data/part-$i-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet; done

# Index(['id', 'url', 'text', 'width', 'height', 'image_phash', 'text_length',
#       'word_count', 'num_tokens_bert', 'num_tokens_gpt', 'num_faces',
#       'clip_similarity_vitb32', 'clip_similarity_vitl14',
#       'nsfw_score_opennsfw2', 'nsfw_score_gantman', 'watermark_score',
#       'aesthetic_score_laion_v2'],
#      dtype='object')
#
img2dataset --url_list /mnt/bn/bytenn-yg2/datasets/coyo_700m/meta_data --input_format "parquet"\
         --url_col "url" --caption_col "text" --output_format webdataset\
           --output_folder /mnt/bn/bytenn-yg2/datasets/coyo_700m/data --processes_count 64 --thread_count 256 --image_size 1024\
            --resize_only_if_bigger=True --resize_mode="keep_ratio" --skip_reencode=True --max_shard_retry 5 \
            --save_additional_columns '["clip_similarity_vitl14","clip_similarity_vitb32","watermark_score","aesthetic_score_laion_v2"]' --enable_wandb False \
            --distributor multiprocessing
