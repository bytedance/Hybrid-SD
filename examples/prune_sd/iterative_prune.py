# ------------------------------------------------------------------------------------
# Copyright 2023–2024 Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import argparse
import time
import csv
# from diffusers import UNet2DConditionModel
import torch
import numpy as np
import random

from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel
from compression.prune_sd.inference_pipeline import InferencePipeline
from compression.prune_sd.prune_utils import calculate_score
from compression.prune_sd.pruner import UnetPruner
# from accelerate.logging import get_logger
import logging

logger = logging.getLogger(__name__)
# logger = get_logger(__name__, log_level="INFO")

def get_file_list_from_csv(csv_file_path):
    file_list = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)        
        next(csv_reader, None) # Skip the header row
        for row in csv_reader: # (row[0], row[1]) = (img name, txt prompt) 
            file_list.append(row)
    return file_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/mnt/bn/bytenn-yg2/pretrained_models/nota-ai--bk-sdm-small")    
    parser.add_argument("--save_dir", type=str, default="./results/debug",
                        help="$save_dir/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--unet_path", type=str, default=None)
    parser.add_argument("--data_list", type=str, default="/mnt/bn/ycq-lq/data/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--prune_resnet_layers", type=str, default=None)
    parser.add_argument("--keep_resnet_ratio", type=float, default=0.75)
    parser.add_argument("--prune_selfattn_layers", type=str, default=None)
    parser.add_argument("--keep_selfattn_heads_ratio", type=float, default=0.75)
    parser.add_argument("--prune_crossattn_layers", type=str, default=None)
    parser.add_argument("--keep_crossattn_heads_ratio", type=float, default=0.75)
    parser.add_argument("--output_type", type=str, default="latent", choices=["pil", "latent"])
    parser.add_argument("--calc_flops", action="store_true")
    parser.add_argument("--n_choices", type=int, default=3)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action='store_true')
    parser.add_argument("--use_dpm_solver", action='store_true')
    parser.add_argument("--use_ddpm_solver", action='store_true')
    parser.add_argument("--iterative_steps", type=int, default=5)
    parser.add_argument("--target_ratio", type=float, default=10)

    args = parser.parse_args()
    return args

def save_output(pipeline, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    imgs = pipeline.generate(prompt = val_prompts,
                            n_steps = args.num_inference_steps,
                            img_sz = args.img_sz)[0]
    imgs.save(os.path.join(save_dir, f"pruned.png"))
    pruner.unet.save_pretrained(os.path.join(save_dir, "unet"))


if __name__ == "__main__":
    args = parse_args()
    logging_dir = args.save_dir
    os.makedirs(logging_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(logging_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    save_dir_src = os.path.join(args.save_dir, f'im{args.img_sz}') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)

    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)
    pipeline.set_pipe_and_generator()   
    unet = UNet2DConditionModel.from_pretrained("nota-ai/bk-sdm-small", subfolder="unet", torch_dtype=torch.float16).to(args.device)
    module_name_dict = {name: module for name, module in unet.named_modules()}
    
    val_prompts = ["A bowl that has vegetables inside of it.",
                   "a brown and white cat staring off with pretty green eyes.",
                   "a golden vase with different flowers."]
    imgs = pipeline.generate(prompt = val_prompts,
                                n_steps = args.num_inference_steps,
                                img_sz = args.img_sz,
                                output_type = 'latent')
    tea_latents = imgs.cpu().detach().numpy()
    example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
    unet.eval()
    pruner = UnetPruner(args, unet, example_inputs=example_inputs)
    org_flops, org_macs, org_params = pruner.calc_flops()
    logger.info(f'Origin params: {org_params}, macs: {org_macs}, flops: {org_flops}')

    pipeline.pipe.unet = pruner.unet
    params_str = pipeline.get_sdm_params()
    
    total_choices = []
    invalid_choices = []
    for n_iter in range(args.iterative_steps):
        iter_choices = []
        latents = [tea_latents]

        for choice in pruner.candidate_choices:
            if choice in invalid_choices:
                continue

            pruner.set_choice(choice)
            success = pruner.prune()
            
            if not success:
                pruner.reset_choice()
                continue

            iter_choices.append(choice)
            imgs = pipeline.generate(prompt = val_prompts,
                                    n_steps = args.num_inference_steps,
                                    img_sz = args.img_sz,
                                    output_type = 'latent')
            
            np_latents = imgs.cpu().detach().numpy()
            latents.append(np_latents)
            pruner.recover()
       
        scores = calculate_score(latents)
        logger.info(f'iter_choices={iter_choices}, scores={scores}')
        top1_idx = np.argsort(scores)[0]
        current_choice = iter_choices[top1_idx]
        total_choices.append(current_choice)
        logger.info(f'top1 idx = {top1_idx}, score = {scores[top1_idx]}, top1 choice = {current_choice}')
        pruner.set_choice(current_choice)
        success = pruner.prune(change_config=True)
        flops, macs, params = pruner.calc_flops()
        ratio_macs = (org_macs - macs) / org_macs * 100
        ratio_params = (org_params - params) / org_params * 100
        logger.info(f'params: {params} ({ratio_params:.2f}), macs: {macs} ({ratio_macs:.2f})\n')
        # import pdb; pdb.set_trace()

        # save_dir = os.path.join(args.save_dir, f"prune-{n_iter}")
        # pipeline.pipe.unet = pruner.unet
        # save_output(pipeline, save_dir)
        # unet = UNet2DConditionModel.from_pretrained(save_dir, subfolder="unet", torch_dtype=torch.float16).to(args.device)
        # pruner.unet = unet
        # print(pruner.unet.down_blocks[3].resnets[0])
        # import pdb; pdb.set_trace()

    logger.info(f'Final choice={total_choices} params: {params} ({ratio_params:.2f}), macs: {macs}\n')
    flops, macs, params = pruner.calc_flops()
    save_output(pipeline, f'{args.save_dir}/latest')
    
