python src/generate_tome.py --model_id /mnt/bn/bytenn-yg2/pretrained_models/Byte_SD1.5_V1 --save_dir results/byte_sd1.5 

python src/generate_tome.py --model_id /mnt/bn/bytenn-yg2/pretrained_models/Byte_SD1.5_V1 --save_dir results/byte_sd1.5_tome_r0.5 --tome --tome_ratio 0.5

python src/generate_tome.py --model_id /mnt/bn/bytenn-yg2/pretrained_models/Byte_SD1.5_V1 --save_dir results/byte_sd1.5_deepcache_i5 --deepcache_cache_internal 5 --deepcache

python src/generate_tome.py --model_id /mnt/bn/bytenn-yg2/pretrained_models/Byte_SD1.5_V1 --save_dir results/byte_sd1.5_deepcache_i3 --deepcache_cache_internal 3 --deepcache

python src/generate_tome.py --model_id /mnt/bn/bytenn-yg2/pretrained_models/Byte_SD1.5_V1 --save_dir results/byte_sd1.5_tgate --tgate
