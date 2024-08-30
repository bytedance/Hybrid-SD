##### Prune Different Modules
export PYTHONPATH='.'
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org 

base_arch=bk-sdm-tiny
DATA_ROOT=/mnt/bn/bytenn-yg2/datasets

# generate latent
python3 examples/prune_sd/get_latents.py --save_dir results/NaivePrune/$base_arch/prune_oneshot --data_list $DATA_ROOT/mscoco_val2014_30k/metadata.csv

