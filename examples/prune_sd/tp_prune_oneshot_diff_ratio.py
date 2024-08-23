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
from compression.prune_sd.models.resnet import ResnetBlock2D
from compression.prune_sd.models.transformer_2d import BasicTransformerBlock
from compression.prune_sd.prune_utils import calculate_score, get_dataloader
# from accelerate.logging import get_logger
import logging
from copy import deepcopy
import pickle
import compression.torch_pruning as tp
from datasets import load_dataset
import torch.nn.functional as F
import math

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
    parser.add_argument("--model_id", type=str, default="/mnt/bn/bytenn-yg2/pretrained_models/nota-ai--bk-sdm-tiny")    
    parser.add_argument("--base_arch", type=str, default="bk-sdm-tiny")    
    parser.add_argument("--save_dir", type=str, default="./results/debug",
                        help="$save_dir/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--unet_path", type=str, default=None)
    parser.add_argument("--data_list", type=str, default="/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_sz", type=int, default=1)
    parser.add_argument("--prune_resnet_layers", type=str, default=None)
    parser.add_argument("--keep_resnet_ratio", type=float, default=0.5)
    parser.add_argument("--prune_selfattn_layers", type=str, default=None)
    parser.add_argument("--keep_selfattn_heads_ratio", type=float, default=0.5)
    parser.add_argument("--prune_crossattn_layers", type=str, default=None)
    parser.add_argument("--keep_crossattn_heads_ratio", type=float, default=0.5)
    parser.add_argument("--output_type", type=str, default="pil", choices=["pil", "latent"])
    parser.add_argument("--calc_flops", action="store_true")
    parser.add_argument("--n_choices", type=int, default=3)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action='store_true')
    parser.add_argument("--use_dpm_solver", action='store_true')
    parser.add_argument("--use_ddpm_solver", action='store_true')
    parser.add_argument("--iterative_steps", type=int, default=5)
    parser.add_argument("--pruning_ratio", type=float, default=0.5)
    parser.add_argument("--pruner", type=str, default='l1', choices=['taylor', 'random', 'l1', 'l2', 'reinit', 'diff-pruning'])
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")
    parser.add_argument("--train_data_dir", type=str, default='/mnt/bn/bytenn-yg2/datasets/laion_aes/preprocessed_11k')
    parser.add_argument("--score_file", type=str, default='results/score-prune/bk-sdm-small-diff-pruning/ratio-0.5/scores.pkl')
    parser.add_argument("--th", type=float, default=0)
    parser.add_argument("--a", type=int, default=10)
    parser.add_argument("--b", type=int, default=20)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    logging_dir = args.save_dir
    os.makedirs(logging_dir, exist_ok=True)
    
    with open(args.score_file, 'rb') as f:
        scores = pickle.load(f)

    for k, v in scores.items():
        if 'resnets' in k:
            scores[k] -= args.th
    scores = dict(sorted(scores.items(), key=lambda item: item[1]))        

    resnet_scores = dict()
    selfattn_scores = dict()
    crossattn_scores = dict()
    total_layers = []
    for k, v in scores.items():
        key = k.split(' ')[0]
        total_layers.append(key)
        if 'resnets' in k:
            resnet_scores[key] = v
        if 'attn1' in k:
            selfattn_scores[key] = v
        if 'attn2' in k:
            crossattn_scores[key] = v

    def get_value(scores):
        np_scores = np.array(scores)
        avg_val = np_scores.mean()
        max_val = np_scores.max()
        min_val = np_scores.min()
        return avg_val, max_val, min_val
        
    avg, max, min = get_value(list(resnet_scores.values()))
    print(f">> resnets = {resnet_scores}, avg_score = {avg:.2f}, max_score = {max:.2f}, min_score = {min:.2f}")
    avg, max, min = get_value(list(selfattn_scores.values()))
    print(f">> selfattn_scores = {selfattn_scores}, avg_score = {avg:.2f}, max_score = {max:.2f}, min_score = {min:.2f}")
    avg, max, min = get_value(list(crossattn_scores.values()))
    print(f">> crossattn_scores = {crossattn_scores}, avg_score = {avg:.2f}, max_score = {max:.2f}, min_score = {min:.2f}")
    resnet_layers = list(resnet_scores.keys())
    selfattn_layers = list(selfattn_scores.keys())
    crossattn_layers = list(crossattn_scores.keys())
 
    if args.base_arch == 'bk-sdm-small':
        block_mid_channels = [320, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 320, 320]
    elif args.base_arch == 'bk-sdm-tiny':
        block_mid_channels = [320, 640, 1280, 1280, 1280, 640, 640, 320, 320]
    else:
        raise NotImplementedError

    selfattn_out_dim = [320, 640, 1280, 1280, 1280, 640, 640, 320, 320]
    crossattn_out_dim = [320, 640, 1280, 1280, 1280, 640, 640, 320, 320]
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(logging_dir, 'log.txt')),
            logging.StreamHandler()
        ]
    )

    if args.pruner == 'taylor':
        imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
    elif args.pruner == 'random' or args.pruner=='reinit':
        imp = tp.importance.RandomImportance()
    elif args.pruner == 'l1':
        imp = tp.importance.MagnitudeImportance(p=1)
    elif args.pruner == 'l2':
        imp = tp.importance.MagnitudeImportance(p=2)
    elif args.pruner == 'diff-pruning':
        imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
    else:
        raise NotImplementedError

    save_dir_src = os.path.join(args.save_dir, f'im{args.img_sz}') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)

    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16).to(args.device)
    
    val_prompts = ["A bowl that has vegetables inside of it.",
                "a brown and white cat staring off with pretty green eyes."]
    example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
    base_macs, base_params = tp.utils.count_ops_and_params(unet, example_inputs)
    print(f"base_macs={base_macs/1e9:.4f} G, base_params={base_params/1e6:.4f} M")

    channel_groups = {}
    unet.zero_grad()
    unet.eval()
    ignored_layers = []
    pruning_ratio_dict = {}
    n_resnet_block, n_attn_block = 0, 0
    
    def get_ratio(name, total_layers, a=10, b=20):
        if name in total_layers[:a]:
            return 0.75
        if name in total_layers[a:b]:
            return 0.5
        if name in total_layers[b:]:
            return 0.25
        
    for name, module in unet.named_modules():
        if 'conv_out' in name or 'proj_out' in name or 'ff.net.2' in name or 'time_embedding' in name or 'ff.net.0.proj' in name or 'conv_shortcut' in name or 'upsamplers' in name or 'downsamplers' in name:
            ignored_layers.append(module)

        if isinstance(module, ResnetBlock2D):
            ratio = get_ratio(name + '.conv1', total_layers, args.a, args.b)
            print(f'{name} ratio={ratio}')
            pruning_ratio_dict[module] = ratio
            
        if isinstance(module, BasicTransformerBlock):
            ratio = get_ratio(name + '.attn1.to_v', total_layers, args.a, args.b)
            pruning_ratio_dict[module.attn1] = ratio
            print(f'{name}.attn1 ratio={ratio}')

            ratio = get_ratio(name + '.attn2.to_v', total_layers, args.a, args.b)
            pruning_ratio_dict[module.attn2] = ratio
            print(f'{name}.attn2 ratio={ratio}')
            
    pruner = tp.pruner.MagnitudePruner(
            unet,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            global_pruning=args.global_pruning,
            channel_groups=channel_groups,
            pruning_ratio_dict=pruning_ratio_dict,
            ignored_layers=ignored_layers,
            prune_num_heads=True
        )

    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)
    pipeline.set_pipe_and_generator() 
    pipeline.pipe.unet = unet

    # loading images for gradient-based pruning
    if args.pruner in ['taylor', 'diff-pruning']:
        logger.info(f"*** load dataset {args.train_data_dir}: start")
        t0 = time.time()
        dataset = load_dataset(
            "imagefolder", 
            data_dir=args.train_data_dir, 
            split="train")
        logger.info(f"*** load dataset: end --- {time.time()-t0} sec")
        train_dataloader = get_dataloader(dataset, pipeline)

        loss_max = 0
        logger.info("Accumulating gradients for pruning...")
        weight_dtype = torch.float16
        for step, batch in enumerate(train_dataloader):
            latents = pipeline.pipe.vae.encode(batch["pixel_values"].to(weight_dtype).to(args.device)).latent_dist.sample()
            latents = latents * pipeline.pipe.vae.config.scaling_factor
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            timesteps = torch.randint(0, pipeline.pipe.scheduler.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            latents = latents.to(args.device)
            noise = noise.to(args.device)
            timesteps = timesteps.to(args.device)
            noisy_latents = pipeline.pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Get the text embedding for conditioning
            encoder_hidden_states = pipeline.pipe.text_encoder(batch["input_ids"].to(args.device))[0]
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss.backward()

            if args.pruner=='diff-pruning':
                if loss > loss_max: loss_max = loss
                if loss < loss_max * args.thr: break # taylor expansion over pruned timesteps ( L_t / L_max > thr )

    for g_idx, g in enumerate(pruner.step(interactive=True)):
        # logger.info(f'Try Pruning block [{g_idx}] {g._group[0].dep.source.name}')
        g.prune()
        # logger.info(f'Pruned module [{g_idx}] {g._group[0].dep.source.name}')
        macs, params = tp.utils.count_ops_and_params(unet, example_inputs)

    ratio_params = (1 - params / base_params) * 100
    ratio_macs = (1 - macs / base_macs) * 100
    logger.info(f"#Params: {base_params/1e6:.4f} M => {params/1e6:.4f} M ({ratio_params:.2f}%), #MACS: {base_macs/1e9:.4f} G => {macs/1e9:.4f} G ({ratio_macs:.2f}%)")

    unet._internal_dict['block_mid_channels'] = block_mid_channels
    unet._internal_dict['selfattn_out_dim'] = selfattn_out_dim
    unet._internal_dict['crossattn_out_dim'] = crossattn_out_dim

    n_resnet_block, n_attn_block = 0, 0
    for name, module in unet.named_modules():
        if isinstance(module, ResnetBlock2D):
            unet._internal_dict['block_mid_channels'][n_resnet_block] = module.conv1.out_channels
            n_resnet_block += 1
        if isinstance(module, BasicTransformerBlock):
            unet._internal_dict['selfattn_out_dim'][n_attn_block] = module.attn1.to_v.out_features
            unet._internal_dict['crossattn_out_dim'][n_attn_block] = module.attn2.to_v.out_features
            n_attn_block += 1

    unet.save_pretrained(os.path.join(args.save_dir, f"unet"))
    imgs = pipeline.generate(prompt = val_prompts,
                                n_steps = args.num_inference_steps,
                                img_sz = args.img_sz,
                                output_type = "pil")
    for img_id, img in enumerate(imgs):
        img.save(os.path.join(save_dir_src, f'img_{img_id}.png'))
    print(f"save result to {save_dir_src}")

    

