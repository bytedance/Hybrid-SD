# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import os
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler, 
    #LCMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionXLPipeline,
)
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from diffusers.utils.import_utils import is_xformers_available


from compression.prune_sd.LCM.Scheduling_LCM import LCMScheduler
#from compression.hybrid_sd.diffusers.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
#from compression.hybrid_sd.diffusers.pipeline_stable_diffusion import StableDiffusionPipeline
# from compression.hybrid_sd.diffusers.pipline_hybrid_LCM import StableDiffusionPipeline

def log_validation(vae, unet, scheduler_path, device, weight_dtype, step=4, seed="1234", save_path=""):
    print("Running validation... ")

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        scheduler_path,
        vae=vae,
        unet=unet,
        torch_dtype=weight_dtype,
    )
    
    scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing")
    scheduler.set_timesteps(12)

    pipeline = pipeline.to(device)
    pipeline.scheduler = scheduler
    
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
    # SDXL
    SDXL_path = "pretrained_models/stabilityai--stable-diffusion-xl-base-1.0"
    save_path = "results/sdxl_base/infer_imgs_sdxl/"

    vae = AutoencoderKL.from_pretrained(
        SDXL_path,
        subfolder="vae",
    )
    # loading SDXL model
    unet = UNet2DConditionModel.from_pretrained(
        SDXL_path, subfolder="unet",
    )
    log_validation(vae, unet, SDXL_path, device="cuda:0", weight_dtype=torch.float32, step=25, seed=1234, save_path=save_path)   
    
    

    