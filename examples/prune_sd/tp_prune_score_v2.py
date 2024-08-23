import os
import argparse
import time
import csv
# from diffusers import UNet2DConditionModel
import torch
import numpy as np
import random
import pickle
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel
from compression.prune_sd.inference_pipeline import InferencePipeline
from compression.prune_sd.prune_utils import calculate_score
from compression.prune_sd.models.resnet import ResnetBlock2D
from compression.prune_sd.models.transformer_2d import BasicTransformerBlock
import logging
from copy import deepcopy
from datasets import load_dataset
import torch.nn.functional as F

import compression.torch_pruning as tp
from compression.prune_sd.prune_utils import get_dataloader

logger = logging.getLogger(__name__)

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
    parser.add_argument("--model_id", type=str, default="/mnt/bn/ycq-lq/hf_models/nota-ai--bk-sdm-small")    
    parser.add_argument("--base_arch", type=str, default="bk-sdm-small") 
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
    parser.add_argument("--layer_pruning_ratio", type=float, default=0.5)
    parser.add_argument("--model_pruning_ratio", type=float, default=0.3)
    parser.add_argument("--pruner", type=str, default='l1', choices=['taylor', 'random', 'l1', 'l2', 'reinit', 'diff-pruning'])
    parser.add_argument("--train_data_dir", type=str, default='/mnt/bn/ycq-lq/data/laion_aes/preprocessed_11k')
    parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

    args = parser.parse_args()
    return args

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

    if args.base_arch == 'bk-sdm-small':
        block_mid_channels = [320, 640, 1280, 1280, 1280, 1280, 1280, 1280, 640, 640, 320, 320]
    elif args.base_arch == 'bk-sdm-tiny':
        block_mid_channels = [320, 640, 1280, 1280, 1280, 640, 640, 320, 320]
    else:
        raise NotImplementedError

    selfattn_out_dim = [320, 640, 1280, 1280, 1280, 640, 640, 320, 320]
    crossattn_out_dim = [320, 640, 1280, 1280, 1280, 640, 640, 320, 320]

    unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet", torch_dtype=torch.float16).to(args.device)
    
    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                args = args)
    pipeline.set_pipe_and_generator()
    pipeline.pipe.unet = unet

    val_prompts = ["A bowl that has vegetables inside of it.",
                "a brown and white cat staring off with pretty green eyes.",
                "a photo of sks11df dog in the snow."
                ]
    imgs = pipeline.generate(prompt = val_prompts,
                                n_steps = args.num_inference_steps,
                                img_sz = args.img_sz,
                                output_type = 'latent')
    tea_latents = imgs.cpu().detach().numpy()

    imgs = pipeline.generate(prompt = val_prompts,
                                n_steps = args.num_inference_steps,
                                img_sz = args.img_sz,
                                output_type = 'pil')
    save_dir_src = os.path.join(args.save_dir, 'baseline', f'im{args.img_sz}') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)
    for img_id, img in enumerate(imgs):
        img_path = os.path.join(save_dir_src, f'img_{img_id}.png')
        img.save(img_path)

    example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(args.device), torch.randn(1, dtype=torch.float16).to(args.device), torch.randn((2, 77, 768), dtype=torch.float16).to(args.device)]
    base_macs, base_params = tp.utils.count_ops_and_params(unet, example_inputs)
    logger.info(f"base_macs={base_macs/1e9:.4f} G, base_params={base_params/1e6:.4f} M")

    n_blocks = len(block_mid_channels) + len(selfattn_out_dim) + len(crossattn_out_dim)
    print(f"total pruning blocks = {n_blocks}")
    max_niters = 100
    top_n = 2
    base_min_channels = dict()
    min_keep_ratio = 0.2
    choices = []

    iter_choices = []
    eval_latents = [tea_latents]

    unet.zero_grad()
    unet.eval()

     # loading images for gradient-based pruning
    if args.pruner in ['taylor', 'diff-pruning']:
        logger.info(f"*** load dataset {args.train_data_dir}: start")
        t0 = time.time()
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir, 
            split="train")
        logger.info(f"*** load dataset: end --- {time.time()-t0} sec")
        train_dataloader = get_dataloader(dataset, pipeline, batch_size=4)

    bound_unet = deepcopy(unet)
    
    for cur_prune_block in range(n_blocks):
        # prune each block
        ignored_layers = []
        for name, module in unet.named_modules():
            if 'conv_out' in name or 'proj_out' in name or 'ff.net.2' in name or 'time_embedding' in name or 'ff.net.0.proj' in name or 'conv_shortcut' in name or 'upsamplers' in name or 'downsamplers' in name:
                ignored_layers.append(module)
                
        pruner = tp.pruner.MagnitudePruner(
                unet,
                example_inputs,
                importance=imp,
                iterative_steps=1,
                pruning_ratio=args.layer_pruning_ratio,
                ignored_layers=ignored_layers,
            )
        
        if args.pruner in ['taylor', 'diff-pruning']:
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
            # print(g)
            if g_idx != cur_prune_block:
                continue

            logger.info(f'Try pruning block [{g_idx}] {g._group[0].dep.source.name}')
            iter_choices.append(g._group[0].dep.source.name)
            g.prune()
            if not g.valid:
                # remove appended item
                iter_choices = iter_choices[:-1]
                continue
            
            logger.info(f'Pruned module [{g_idx}] {g._group[0].dep.source.name}, valid={g.valid}')
            pipeline.pipe.unet = unet
            imgs = pipeline.generate(prompt = val_prompts,
                                    n_steps = args.num_inference_steps,
                                    img_sz = args.img_sz,
                                    output_type = 'latent')
            
            np_latents = imgs.cpu().detach().numpy()
            eval_latents.append(np_latents)

            imgs = pipeline.generate(prompt = val_prompts,
                                    n_steps = args.num_inference_steps,
                                    img_sz = args.img_sz,
                                    output_type = 'pil')

            prune_layer_name = g._group[0].dep.source.name.split(' ')[0]

            save_dir = os.path.join(args.save_dir, f'{prune_layer_name}', f'img_{img_id}.png')
            os.makedirs(save_dir, exist_ok=True)
            for img_id, img in enumerate(imgs):
                img_path = os.path.join(save_dir, f'img_{img_id}.png')
                img.save(img_path)
                
            del unet
            del pruner
            unet = deepcopy(bound_unet)
            unet.zero_grad()
            unet.eval()
            
    # get top-n least important modules
    scores = calculate_score(eval_latents)
    choice_score_mapping = {k: v for k, v in zip(iter_choices, scores)}
    choice_score_mapping = dict(sorted(choice_score_mapping.items(), key=lambda item: item[1]))
    print(f"save score file to {args.save_dir}")

    with open(os.path.join(args.save_dir, 'scores.pkl'), 'wb') as f:
        pickle.dump(choice_score_mapping, f)

