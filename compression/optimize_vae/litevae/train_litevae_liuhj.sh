#!/bin/bash
GPUS=1
NNODES=1
NODE_RANK=${RANK:-0}
# PORT=${PORT:-2333}
PORT=23333
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}   
OUTDIR="/mnt/bn/bytenn-data2/liuhj/experiment/vae/litevae"

NCCL_IB_HCA=`ibdev2netdev|awk '{print$1}'`
roce_PORT=":1"
NCCL_IB_HCA=${NCCL_IB_HCA}${roce_PORT}
NCCL_DEBUG=TRACE
OMPI_MCA_btl_tcp_if_include=eth0
NCCL_SOCKET_IFNAME=eth0
NCCL_IB_DISABLE=0
NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA
export NCCL_DEBUG
export OMPI_MCA_btl_tcp_if_include
export NCCL_SOCKET_IFNAME
export NCCL_IB_DISABLE
export NCCL_IB_GID_INDEX  
export /mnt/bn/bytenn-data2/liuhj/pylib:$PYTHONPATH
export WANDB_DIR=$OUTDIR
# pip3 install prefetch_generator  diffusers==0.27.0 huggingface_hub -i https://mirrors.aliyun.com/pypi/simple/. #xformers huggingface_hub==0.19.0
torchrun --nnodes=$NNODES --master_addr=$MASTER_ADDR  --master_port=$PORT --node_rank=$NODE_RANK --nproc_per_node=$GPUS train_litevae_liuhj.py --MASTER ${MASTER_ADDR}  \
--output_dir $OUTDIR --learning_rate 4.5e-06 --lr_scheduler "constant" --lr_warmup_steps 0 --adam_weight_decay 0.01 --seed 42 --gradient_accumulation_steps 1 \
--checkpointing_steps 2000 --train_batch_size 1 --num_train_epochs 2000 --resolution 128 --mixed_precision no  \ #--report_to "wandb" \
--pretrained_model_name_or_path "/mnt/bn/bytenn-data2/sd_models/runwayml--stable-diffusion-v1-5" | tee logs/ddp-coco-cm_lion_sde.log

