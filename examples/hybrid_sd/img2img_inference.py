import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import os
import sys
sys.path.insert(0,"/usr/local/lib/python3.9/dist-packages")
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    #LCMScheduler,
    #StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    UNet2DConditionModel,
)
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available
from PIL import Image

from compression.prune_sd.LCM.Scheduling_LCM import LCMScheduler
#from compression.hybrid_sd.diffusers.pipeline_stable_diffusion import StableDiffusionPipeline
from compression.hybrid_sd.diffusers.pipline_hybrid_LCM import StableDiffusionPipeline

def log_validation(vae, unet, sd_path, device, weight_dtype, imgs_path, seed="1234", save_path=""):
    print("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        sd_path,
        vae=vae,
        unet=unet,
        torch_dtype=weight_dtype,
    )

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)


    pipeline.enable_xformers_memory_efficient_attention()

    if seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    validation_prompts = [
        "more flowers",
        "a bird standing on the stage"
    ]

    image_logs = []
    imgs = Image.open(imgs_path)
    

    os.makedirs(save_path, exist_ok=True)
    for _, prompt in enumerate(validation_prompts):
        images = []

        autocast_ctx = torch.autocast(device_type="cuda", dtype=weight_dtype)

        with autocast_ctx:
            images = pipeline(
                prompt=prompt,
                image=imgs,
                num_images_per_prompt=1,
                guidance_scale=7.5,
                strength=0.75,
                generator=generator,
            ).images
        image_logs.append({"validation_prompt": prompt, "images": images})
        for img in images:
            img.save(save_path + prompt + '.png')
    return image_logs
    
if __name__=="__main__":
    # SD1.5
    # SD15_path = "/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5"
    # vae = AutoencoderKL.from_pretrained(
    #     SD15_path,
    #     subfolder="vae",
    # )
    # # loading SD1.5 LCM model
    # LCM_SD_path = "results/lcm_sd15_2w/checkpoint-20000"
    # save_path = "results/lcm_sd15_2w/infer_imgs/"
    # unet = UNet2DConditionModel.from_pretrained(
    #     LCM_SD_path, subfolder="unet",
    # )
    # log_validation(vae, unet, SD15_path, device="cuda:0", weight_dtype=torch.float32, step=8, seed=1234, save_path=save_path)
    
    
    # SD1.4 
    save_path = "results/sd14/img2img/"
    SD14_path = "/mnt/bn/bytenn-yg2/pretrained_models/CompVis--stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(
        SD14_path,
        subfolder="vae",
    )

    unet = UNet2DConditionModel.from_pretrained(
        SD14_path, subfolder="unet",
    )
    # TODO imgs_path
    imgs_path = "/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/results/debug/im512/img_3.png"
    log_validation(vae, unet, SD14_path, device="cuda:0", weight_dtype=torch.float32,  imgs_path=imgs_path, seed=1234, save_path=save_path)   
    
    
    # loading Tiny model
    # SD14_path = "/mnt/bn/bytenn-yg2/pretrained_models/CompVis--stable-diffusion-v1-4"
    # vae = AutoencoderKL.from_pretrained(
    #     SD14_path,
    #     subfolder="vae",
    # )
    # Tiny_LCM_path = "results/nota-ai--bk-sdm-tiny_LCM/checkpoint-20000"
    # save_path = "results/nota-ai--bk-sdm-tiny_LCM/infer_imgs/"
    # unet = CustomUNet2DConditionModel.from_pretrained(
    #     Tiny_LCM_path, subfolder="unet_target" #, revision=args.non_ema_revision
    # )
    
    # log_validation(vae, unet, SD14_path, device="cuda:0", weight_dtype=torch.float32, step=8, seed=1234, save_path=save_path)
    