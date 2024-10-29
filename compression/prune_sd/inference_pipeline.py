# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://huggingface.co/blog/stable_diffusion


import diffusers
from diffusers import StableDiffusionPipeline
import torch
import gc
from typing import Union, List
from PIL import Image
from diffusers.utils.import_utils import is_xformers_available
from packaging import version

diffusers_version = int(diffusers.__version__.split('.')[1])

class InferencePipeline:
    def __init__(self, weight_folder, seed, device, args):
        self.weight_folder = weight_folder
        self.seed = seed
        self.device = torch.device(device)
        self.args = args

        self.pipe = None
        self.generator = None

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        if diffusers_version == 15: # for the specified version in requirements.txt
            self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                torch_dtype=torch.float16).to(self.device)
            self.pipe.safety_checker = lambda images, clip_input: (images, False) 
        elif diffusers_version >= 19: # for recent diffusers versions
            self.pipe = StableDiffusionPipeline.from_pretrained(self.weight_folder,
                                                                safety_checker=None, torch_dtype=torch.float16).to(self.device)
        else: # for the versions between 0.15 and 0.19, the benchmark scores are not guaranteed.
            raise Exception(f"Use diffusers version as either ==0.15.0 or >=0.19 (from current {diffusers.__version__})")

        self.pipe.set_progress_bar_config(disable=True)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                self.pipe.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.args.use_dpm_solver:    
            self.args.logger.log(" ** replace PNDM scheduler into DPM-Solver")
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config) 
            
        if self.args.use_ddpm_solver:
            self.args.logger.log(" ** replace PNDM scheduler into DDPM scheduler")
            from diffusers import DDPMScheduler
            self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config) 
            

    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, n_steps: int = 25, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt: int = 1, output_type = "pil"):
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            height = img_sz,
            width = img_sz,
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            output_type = output_type
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_sdm_params(self):
        params_unet = self._count_params(self.pipe.unet)
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        params_total = params_unet + params_text_enc + params_image_dec
        return f"Total {(params_total/1e6):.1f}M (U-Net {(params_unet/1e6):.1f}M; TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"

