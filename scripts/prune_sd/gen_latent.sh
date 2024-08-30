##### Prune Different Modules
export PYTHONPATH='.'


base_arch=bk-sdm-tiny
DATA_ROOT=datasets
MODEL_PATH=pretrained_models/nota-ai--bk-sdm-tiny


# generate latent
python3 examples/prune_sd/get_latents.py --model_id $MODEL_PATH --save_dir results/NaivePrune/$base_arch/prune_oneshot --data_list $DATA_ROOT/mscoco_val2014_30k/metadata.csv

