import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import os
import sys
sys.path.insert(0, "/usr/local/lib/python3.9/dist-packages")
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    #LCMScheduler,
    #StableDiffusionPipeline,
    UNet2DConditionModel,
)
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available

from compression.prune_sd.LCM.Scheduling_LCM import LCMScheduler
#from compression.hybrid_sd.diffusers.pipeline_stable_diffusion import StableDiffusionPipeline
from compression.hybrid_sd.diffusers.pipline_hybrid_LCM import StableDiffusionPipeline
from compression.prune_sd.calflops import calculate_flops

def log_validation(vae, unet, scheduler_path, device, weight_dtype, step=4, seed="1234", save_path=""):
    print("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        scheduler_path,
        vae=vae,
        unet=unet,
        scheduler=LCMScheduler.from_pretrained(scheduler_path, subfolder="scheduler"),
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
        "portrait photo of a girl, photograph, highly detailed face, depth of field, moody light, golden hour, style by Dan Winters, Russell James, Steve McCurry, centered, extremely detailed, Nikon D850, award winning photography",
        "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
        "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
        "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
        "A yellow taxi cab sitting below tall buildings",
        "The shiny motorcycle has been put on display",
        "a professional photograph of an astronaut riding a horse, 8k",
        "muscular warrior fighting giant snake in the lake"
    ]

    image_logs = []
    

    os.makedirs(save_path, exist_ok=True)
    for _, prompt in enumerate(validation_prompts):
        images = []

        autocast_ctx = torch.autocast(device_type="cuda", dtype=weight_dtype)

        with autocast_ctx:
            images = pipeline(
                prompt=prompt,
                num_inference_steps=step,
                num_images_per_prompt=4,
                guidance_scale=7.5,
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
    SD14_path = "/mnt/bn/bytenn-yg2/pretrained_models/CompVis--stable-diffusion-v1-4"
    vae = AutoencoderKL.from_pretrained(
        SD14_path,
        subfolder="vae",
    )
    # loading SD1.4 LCM model
    LCM_SD_path = "results/lcm_sd14_2w/checkpoint-20000"

    save_path = "results/lcm_sd14_2w/infer_imgs/"
    unet = UNet2DConditionModel.from_pretrained(
        LCM_SD_path, subfolder="unet",
    )
    # unet  = UNet2DConditionModel.from_pretrained(
    #     SD14_path, subfolder="unet",
    # )
    #log_validation(vae, unet, SD14_path, device="cuda:0", weight_dtype=torch.float32, step=8, seed=1234, save_path=save_path)   
    device = "cuda:0"
    unet = unet.to(torch.float16).to(device)
    example_inputs = [torch.randn((2, 4, 64, 64), dtype=torch.float16).to(device), torch.randn(1, dtype=torch.float16).to(device), torch.randn((2, 77, 768), dtype=torch.float16).to(device)]
    flops, macs, params = calculate_flops(
                model=unet, 
                args=example_inputs,
                output_as_string=False,
                output_precision=4,
                print_detailed=False,
                print_results=False)
    model_name = "SD1.4"
    model_info = {
        'params': params,
        'flops': flops,
        'macs': macs,
        'total_macs': macs * 8
    }
    total_macs = macs * 8
    print(f'Model{model_name}, #Params: {params/1e6:2f} M, MACs: {macs/1e9:.2f} G, Total Macs: {total_macs/1e12:.2f} T')
    
    import pdb;pdb.set_trace()
    
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
    