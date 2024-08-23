export PYTHONPATH='.'

pruner="diff-pruning"
base_arch="bk-sdm-tiny"
ratio=0.5

GPU_NUM=1

# CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/tp_prune_score_v2.py --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --pruner diff-pruning --save_dir results/score-prune/${base_arch}-diff-pruning --train_data_dir /mnt/bn/bytenn-yg2/datasets/laion_aes/preprocessed_11k

a=5
b=15
th=0
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-diff-pruning/scores.pkl --pruner diff-pruning

a=5
b=20
th=0
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-diff-pruning/scores.pkl --pruner diff-pruning

a=10
b=20
th=0
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-diff-pruning/scores.pkl --pruner diff-pruning

a=15
b=20
th=0
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-diff-pruning/scores.pkl --pruner diff-pruning

a=15
b=30
th=10
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-diff-pruning/scores.pkl --pruner diff-pruning

a=30
b=30
th=0
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/prune_sd/tp_prune_oneshot_diff_ratio.py --a $a --b $b --th $th --save_dir results/score-prune/$base_arch/a${a}_b${b}_th${th} --base_arch $base_arch --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --score_file results/score-prune/$base_arch-diff-pruning/scores.pkl --pruner diff-pruning
