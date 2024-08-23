export PYTHONPATH='.'

pruner="l1"
base_arch="bk-sdm-small"
layer_pruning_ratio=0.25

python3 examples/prune_sd/tp_prune_score_v2.py --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --pruner $pruner --save_dir results/score-prune/$base_arch-$pruner --layer_pruning_ratio $layer_pruning_ratio

# a=5
# b=20
# th=0
# CUDA_VISIBLE_DEVICES=0 python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch-$pruner/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-$pruner/ --pruner $pruner

# a=10
# b=20
# th=0
# CUDA_VISIBLE_DEVICES=0 python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch-$pruner/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-$pruner/ --pruner $pruner

# a=15
# b=20
# th=0
# CUDA_VISIBLE_DEVICES=0 python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch-$pruner/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-$pruner/ --pruner $pruner

# a=15
# b=20
# th=10
# CUDA_VISIBLE_DEVICES=0 python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch-$pruner/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-$pruner/ --pruner $pruner

# a=30
# b=30
# th=0
# CUDA_VISIBLE_DEVICES=0 python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch-$pruner/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-$pruner/ --pruner $pruner

