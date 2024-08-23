##### Prune Different Modules
export PYTHONPATH='.'
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 

base_arch=bk-lcm-tiny
DATA_ROOT=/mnt/bn/bytenn-yg2/datasets

python3 examples/prune_sd/get_latents.py --save_dir results/NaivePrune/$base_arch/prune_oneshot --data_list $DATA_ROOT/mscoco_val2014_30k/metadata.csv


# prune_resnet_layers=(1 2 3 4 5 6 7 8 9 10)
# for layer_id in ${prune_resnet_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py --prune_resnet_layers $layer_id --save_dir results/NaivePrune/$base_arch/prune_resnet_ratio0.5/prune_resnet_$layer_id --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch
# done

# prune_selfattn_layers=(1 2 3 4 5 6 7 8 9)
# for layer_id in ${prune_selfattn_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py ]--prune_selfattn_layers $layer_id --save_dir results/NaivePrune/$base_arch/prune_selfatt_heads4/prune_selfatt_$layer_id --keep_selfattn_heads 4 --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch
# done

# prune_crossattn_layers=(1 2 3 4 5 6 7 8 9)
# for layer_id in ${prune_crossattn_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py --prune_crossattn_layers $layer_id --save_dir results/NaivePrune/$base_arch/prune_crossatt_heads4/prune_crossatt_$layer_id --keep_crossattn_heads 4 --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch
# done

# prune_crossattn_layers=(1 2 3 4 5 6 7 8 9)
# for layer_id in ${prune_crossattn_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py --prune_crossattn_layers $layer_id --save_dir results/NaivePrune/$base_arch/prune_crossatt_heads2/prune_crossatt_$layer_id --keep_crossattn_heads 2 --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch
# done


# a=5
# b=25
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined_v2/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch --a $a --b $b 

# a=10
# b=20
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined_v2/a${a}_b${b} --score_file results/NaivePrune/$base_arch/prune_oneshot/score.pkl --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch --a $a --b $b 

# a=15
# b=25
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined_v2/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch --a $a --b $b 

# a=20
# b=25
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined_v2/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch --a $a --b $b 

# a=3
# b=6
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined_v2/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch --a $a --b $b 
