import argparse
import os

import torch
import numpy as np
import matplotlib.pyplot as plt


from compression.hybrid_sd.tgate.SD_tgate import TgateSDLoader
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from diffusers import PixArtAlphaPipeline,StableDiffusionXLPipeline,StableDiffusionPipeline
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of TGATE.")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="the input prompts",
    )
    parser.add_argument(
        "--saved_path",
        type=str,
        default=None,
        required=True,
        help="Path to save the generated results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default='sd_1.5',
        help="[pixart,sd_xl,sd_2.1,sd_1.5,lcm_sdxl,lcm_pixart,bk_sdm]",
    )
    parser.add_argument(
        "--gate_step",
        type=int,
        default=10,
        help="When re-using the cross-attention",
    )
    parser.add_argument(
        "--inference_step",
        type=int,
        default=25,
        help="total inference steps",
    )
    parser.add_argument(
        '--deepcache', 
        action='store_true', 
        default=False, 
        help='do deep cache',
    )
    
    args = parser.parse_args()
    return args


"""
python3  compression/hybrid_sd/tgate/taget_sd_inference.py  --model sd_1.5 --prompt  "A yellow taxi cab sitting below tall buildings" --saved_path "."
    
python3  compression/hybrid_sd/tgate/taget_sd_inference.py  --model bk_sdm --prompt  "A yellow taxi cab sitting below tall buildings" --saved_path "."

python3  compression/hybrid_sd/tgate/taget_sd_inference.py  --model sd_1.5 --prompt  "A yellow taxi cab sitting below tall buildings" --saved_path "."  --gate_step 50
"""



def calculate_stats(tensor_list):
    tensors = torch.stack(tensor_list)
    mean = torch.mean(tensors).cpu().numpy()
    var = torch.var(tensors).cpu().numpy()
    return mean, var


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.saved_path, exist_ok=True)
    saved_path = os.path.join(args.saved_path, 'test.png')
    
    unet = None
    if args.model in ['sd_2.1', 'sd_1.5','bk_sdm']:
        if args.model == 'sd_1.5':
            repo_id = "/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5"
        elif args.model == 'sd_2.1':
            repo_id = "stabilityai/stable-diffusion-2-1"
        elif args.model == 'bk_sdm':
            repo_id = "/mnt/bn/bytenn-yg2/pretrained_models/nota-ai--bk-sdm-tiny"
            # unet = CustomUNet2DConditionModel.from_pretrained(
            #     repo_id, subfolder="unet_target" #, revision=args.non_ema_revision
            # )
        pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

        

        pipe = TgateSDLoader(pipe)
        pipe = pipe.to("cuda")

        image = pipe.tgate(args.prompt,
                        num_inference_steps=args.inference_step,
                        guidance_scale=7.5,
                        gate_step=args.gate_step,
                        ).images[0]


    image.save(saved_path)