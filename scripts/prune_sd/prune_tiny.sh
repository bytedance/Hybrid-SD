##### Prune Different Modules
export PYTHONPATH='.'

base_arch=bk-sdm-tiny
DATA_ROOT=/mnt/bn/bytenn-yg2/datasets



# a, b means using different pruning ratios for layer_id
# we use a,b to split layer_ids into (0, a), (a, b) and (b, last_layer) and assign them different pruning ratios

a=19
b=21
python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_oneshot/a${a}_b${b} --score_file results/NaivePrune/$base_arch/prune_oneshot/score.pkl --model_id /mnt/bn/bytenn-yg2/pretrained_models/nota-ai--$base_arch --base_arch $base_arch --a $a --b $b 

