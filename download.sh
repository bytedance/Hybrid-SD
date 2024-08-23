# c=$pwd
# /mnt/bd/bytenn-lq
# mkdir laion2B-en-aesthetic && cd laion2B-en-aesthetic
# for i in {00000..00127}; do wget https://huggingface.co/datasets/laion/laion2B-en-aesthetic/resolve/main/part-$i-9230b837-b1e0-4254-8b88-ed2976e9cee9-c000.snappy.parquet; done
# cd ${c}

export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org


path_pretrained_models=/mnt/bn/bytenn-data2/sd_models

# huggingface-cli download --resume-download  fantasyfish/laion-art --local-dir /mnt/bd/bytenn-lq2/datasets/  --local-dir-use-symlinks False --token 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

# huggingface-cli download CompVis/stable-diffusion-v1-4 --local-dir ./pretrained_models/CompVis--stable-diffusion-v1-4  --local-dir-use-symlinks False --token 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



# huggingface-cli download SG161222/Realistic_Vision_V5.1_noVAE --local-dir pretrained_models/SG161222--Realistic_Vision_V5.1_noVAE --local-dir-use-symlinks False --token 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


# huggingface-cli download nota-ai/bk-sdm-tiny --local-dir pretrained_models/nota-ai--bk-sdm-tiny  --local-dir-use-symlinks False --token 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

# huggingface-cli download PixArt-alpha/PixArt-XL-2-1024-MS --local-dir $path_pretrained_models/PixArt-XL-2-1024-MS --local-dir-use-symlinks False --token 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

# huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir $path_pretrained_models/runwayml--stable-diffusion-v1-5  --local-dir-use-symlinks False --token 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

# huggingface-cli download naclbit/trinart_characters_19.2m_stable_diffusion_v1 --local-dir $path_pretrained_models/naclbit--trinart_characters_19.2m_stable_diffusion_v1 --local-dir-use-symlinks False --token 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

huggingface-cli download madebyollin/sdxl-vae-fp16-fix --local-dir /mnt/bn/bytenn-yg2/pretrained_models --local-dir-use-symlinks False --token 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   --resume-download
