##### Prune Different Modules
export PYTHONPATH='.'

base_arch=bk-sdm-tiny
DATA_ROOT=datasets

# step 1. Analyze the significance score of each layer of the U-Net
python3 examples/prune_sd/get_latents.py --model_id $MODEL_PATH --save_dir results/NaivePrune/$base_arch/prune_oneshot --data_list $DATA_ROOT/mscoco_val2014_30k/metadata.csv

# step 2. Calculate the score of each candidate pruning layer based on the predicted latents
python3 examples/prune_sd/analyze_score.py

# step 3. Prune the U-Net based on the calculated score.pkl

# a, b means using different pruning ratios for layer_id
# we use a,b to split layer_ids into (0, a), (a, b) and (b, last_layer) and assign them different pruning ratios

a=19
b=21
python3 examples/prune_sd/naive_prune_oneshot_diff_ratio.py --save_dir results/NaivePrune/$base_arch/prune_oneshot/a${a}_b${b} --score_file results/NaivePrune/$base_arch/prune_oneshot/score.pkl --model_id pretrained_models/nota-ai--$base_arch --base_arch $base_arch --a $a --b $b 