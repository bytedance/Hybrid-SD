##### Prune Different Modules
base_arch="bk-sdm-small"

# python3 examples/prune_sd/naive_prune.py --save_dir results/NaivePrune/$base_arch/prune_oneshot/ratio-0.5 --prune_resnet_layers "1,2,3,4,5,6,7,8,9,10,11,12" --prune_selfattn_layers "1,2,3,4,5,6,7,8,9" --prune_crossattn_layers "1,2,3,4,5,6,7,8,9"

# python3 examples/prune_sd/naive_prune.py --save_dir results/NaivePrune/bk-sdm-small/origin

# prune_resnet_layers=(1 2 3 4 5 6 7 8 9 10 11 12)
# for layer_id in ${prune_resnet_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py --prune_resnet_layers $layer_id --save_dir results/NaivePrune/bk-sdm-small/prune_resnet_ratio0.5/prune_resnet_$layer_id
# done

# prune_selfattn_layers=(1 2 3 4 5 6 7 8 9)
# for layer_id in ${prune_selfattn_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py --prune_selfattn_layers $layer_id --save_dir results/NaivePrune/bk-sdm-small/prune_selfatt_heads4/prune_selfatt_$layer_id --keep_selfattn_heads 4
# done

# prune_crossattn_layers=(1 2 3 4 5 6 7 8 9)
# for layer_id in ${prune_crossattn_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py --prune_crossattn_layers $layer_id --save_dir results/NaivePrune/bk-sdm-small/prune_crossatt_heads4/prune_crossatt_$layer_id --keep_crossattn_heads 4
# done

# prune_crossattn_layers=(1 2 3 4 5 6 7 8 9)
# for layer_id in ${prune_crossattn_layers[@]};
# do
#     python3 examples/prune_sd/naive_prune.py --prune_crossattn_layers $layer_id --save_dir results/NaivePrune/bk-sdm-small/prune_crossatt_heads2/prune_crossatt_$layer_id --keep_crossattn_heads 2
# done


# python3 examples/prune_sd/prune.py --prune_resnet_layers 4,5,7,11 $layer_id --save_dir results/NaivePrune/bk-sdm-small/prune_combined/v1 --keep_crossattn_heads 2 --prune_crossattn_layers 1,6,8,9


# a=5
# b=10
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

# a=5
# b=20
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

# a=5
# b=15
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

# a=10
# b=20
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

# a=10
# b=25
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

# a=10
# b=30
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

# a=15
# b=20
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

# a=15
# b=25
# python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b

a=15
b=28
python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_combined_v2/a${a}_b${b} --score_file results/NaivePrune/$base_arch/score.pkl --a $a --b $b
