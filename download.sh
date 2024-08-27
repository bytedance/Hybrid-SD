# c=$pwd
# /mnt/bd/bytenn-lq
# mkdir laion2B-en-aesthetic && cd laion2B-en-aesthetic
# for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en-aesthetic/resolve/main/part-$i-9230b837-b1e0-4254-8b88-ed2976e9cee9-c000.snappy.parquet; done
# cd ${c}

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org


path_pretrained_models=/mnt/bn/bytenn-data2/sd_models

huggingface-cli download madebyollin/sdxl-vae-fp16-fix --local-dir /mnt/bn/bytenn-yg2/pretrained_models --local-dir-use-symlinks False --token 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   --resume-download
