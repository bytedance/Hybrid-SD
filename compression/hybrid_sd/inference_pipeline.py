# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import torch
import gc
import json
import os 
from peft import LoraModel, LoraConfig, set_peft_model_state_dict
from typing import Union, List
from PIL import Image
from packaging import version
import diffusers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

from diffusers.utils.import_utils import is_xformers_available
from .diffusers.pipeline_stable_diffusion import StableDiffusionPipeline, HybridStableDiffusionPipeline
from .diffusers.pipeline_stable_diffusion_xl import  HybridStableDiffusionXLPipeline
from .diffusers.pipline_hybrid_LCM import HybridLCMPipeline
from compression.prune_sd.models.unet_2d_condition import UNet2DConditionModel as CustomUNet2DConditionModel
from compression.prune_sd.LCM.Scheduling_LCM import LCMScheduler
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

    def set_lora_ckpt(self): 
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            load_and_set_lora_ckpt(
                               pipe=self.pipe,
                               weight_path=os.path.join(self.args.lora_weight_path, 'lora.pt'),
                               config_path=os.path.join(self.args.lora_weight_path, 'lora_config.json'),
                               dtype=torch.float16)

    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, n_steps: int = 25, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt: int = 1, save_path: str = None) -> List[Image.Image]:
        out = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=n_steps,
            height = img_sz,
            width = img_sz,
            generator=self.generator,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            save_path=save_path
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


class HybridInferencePipeline:
    def __init__(self, weight_folders, seed, device, args):
        self.weight_folders = weight_folders
        self.device = torch.device(device)
        self.seed = seed
        self.args = args
        self.args.vae_path = None
        self.pipe = None
        self.generator = None

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        text_encoder = CLIPTextModel.from_pretrained(
                self.weight_folders[0], subfolder="text_encoder"
            ).to(self.device, dtype=torch.float16).requires_grad_(False)
        if self.args.vae_path is not None:
            from diffusers import AutoencoderTiny
            print("loading Tiny VAE")
            vae = AutoencoderTiny.from_pretrained(self.args.vae_path).to(self.device, dtype=torch.float16).requires_grad_(False)
        else:
            if 'Realistic_Vision' in self.weight_folders[0]:
                print("loading stabilityai/sd-vae-ft-ema...")
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device, dtype=torch.float16).requires_grad_(False)
            else:
                vae = AutoencoderKL.from_pretrained(
                        self.weight_folders[0], subfolder="vae"
                    ).to(self.device, dtype=torch.float16).requires_grad_(False)
        tokenizer = CLIPTokenizer.from_pretrained(
                    self.weight_folders[0], subfolder="tokenizer"
            )
        unets = []
        for path in self.weight_folders:
            if 'hybrid-sd' in path:
                MODEL_OBJ = CustomUNet2DConditionModel
            else:
                MODEL_OBJ = UNet2DConditionModel

            unets.append(
                MODEL_OBJ.from_pretrained(
                    path, subfolder="unet"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
            )

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                for unet in unets:
                    unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.pipe = HybridStableDiffusionPipeline.from_pretrained(
            self.weight_folders[0],
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unets[0]
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unets = unets
        total_step, step_config = self.get_step_config(self.args)
        print(f'total_step={total_step}, step_config={step_config}')
        self.pipe.step_config = step_config
        self.total_step = total_step 
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
    
        if self.args.use_dpm_solver:    
            # self.args.logger.info(" ** Use DPMSolverMultistepScheduler")
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)        

        if self.args.use_pndm_solver:    
            # self.args.logger.info(" ** Use PNDMScheduler")
            from diffusers import PNDMScheduler
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)        
        
        # skip safety_checker
        self.pipe.safety_checker = None

    def set_lora_ckpt(self):
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            hybrid_load_and_set_lora_ckpt(
                                pipe=self.pipe,
                                weight_path=[os.path.join(path, 'lora.pt') for path in self.args.lora_weight_path],
                                config_path=[os.path.join(path, 'lora_config.json') for path in self.args.lora_weight_path],
                                dtype=torch.float16)


    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, save_path=None,prompt_embeds=None, negative_prompt_embeds=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path,
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_step_config(self, args):
        assert len(args.steps) > 0 
        total_step = sum(args.steps)
        assert total_step > 0
        assert len(self.weight_folders) == len(args.steps)
        step_config = {
            "step":{},
            "name":{}
        }
        total_step = 0 
        for index, model_step in enumerate(args.steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        for index, model_name in enumerate(self.weight_folders):
            step_config["name"][index] = model_name.split("/")[-1]

        return total_step, step_config

    def get_sdm_params(self):
        params_str = ""
        for index in range(len(self.pipe.unets)):
            model_name = self.weight_folders[index].split("/")[-1]
            cur_unet = self._count_params(self.pipe.unets[index])
            params_str += f" {model_name}: {(cur_unet/1e6):.1f}M"
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        return_str =  params_str + f"TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"
        return return_str


class HybridSDXLInferencePipeline:
    def __init__(self, weight_folders, seed, device, args):
        self.weight_folders = weight_folders
        self.device = torch.device(device)
        self.seed = seed
        self.args = args
        self.args.vae_path = None
        self.pipe = None
        self.generator = None
        if args.weight_dtype == "fp16":
            self.weight_dtype = torch.float32 
        elif args.weight_dtype == "fp16":
            self.weight_dtype = torch.float16

    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation**         
        if self.args.vae_path is not None:
            from diffusers import AutoencoderTiny
            print("loading Tiny VAE")
            vae = AutoencoderTiny.from_pretrained(self.args.vae_path).to(self.device, dtype=torch.float16).requires_grad_(False)
        else:
            vae = AutoencoderKL.from_pretrained(
                    "pretrained_models/madebyollin--sdxl-vae-fp16-fix", subfolder="vae"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
        unets = []
        for path in self.weight_folders:
            MODEL_OBJ = UNet2DConditionModel
            unets.append(
                MODEL_OBJ.from_pretrained(
                    path, subfolder="unet"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
            )

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                for unet in unets:
                    unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.pipe = HybridStableDiffusionXLPipeline.from_pretrained(
            self.weight_folders[0],
            vae=vae,
            unet=unets[0],
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unets = unets
        total_step, step_config = self.get_step_config(self.args)
        print(f'total_step={total_step}, step_config={step_config}')
        self.pipe.step_config = step_config
        self.total_step = total_step 
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.pipe.to(self.device)
    
        if self.args.use_dpm_solver:    
            # self.args.logger.info(" ** Use DPMSolverMultistepScheduler")
            from diffusers import DPMSolverMultistepScheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)        

        if self.args.use_pndm_solver:    
            # self.args.logger.info(" ** Use PNDMScheduler")
            from diffusers import PNDMScheduler
            self.pipe.scheduler = PNDMScheduler.from_config(self.pipe.scheduler.config)        
        
        # skip safety_checker
        self.pipe.safety_checker = None

    def set_lora_ckpt(self):
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            hybrid_load_and_set_lora_ckpt(
                                pipe=self.pipe,
                                weight_path=[os.path.join(path, 'lora.pt') for path in self.args.lora_weight_path],
                                config_path=[os.path.join(path, 'lora_config.json') for path in self.args.lora_weight_path],
                                dtype=torch.float16)


    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, save_path=None, prompt_embeds=None, negative_prompt_embeds=None, image_prompt_embeds=None, uncond_image_prompt_embeds=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path,
            image_prompt_embeds=image_prompt_embeds,
            uncond_image_prompt_embeds=uncond_image_prompt_embeds,
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_step_config(self, args):
        assert len(args.steps) > 0 
        total_step = sum(args.steps)
        assert total_step > 0
        assert len(self.weight_folders) == len(args.steps)
        step_config = {
            "step":{},
            "name":{}
        }
        total_step = 0 
        for index, model_step in enumerate(args.steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        for index, model_name in enumerate(self.weight_folders):
            step_config["name"][index] = model_name.split("/")[-1]

        return total_step, step_config

    def get_sdm_params(self):
        params_str = ""
        for index in range(len(self.pipe.unets)):
            model_name = self.weight_folders[index].split("/")[-1]
            cur_unet = self._count_params(self.pipe.unets[index])
            params_str += f" {model_name}: {(cur_unet/1e6):.1f}M"
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        return_str =  params_str + f"TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"
        return return_str




class HybridLCMInferencePipeline:
    def __init__(self, weight_folders, seed, device, args):
        self.weight_folders = weight_folders
        self.device = torch.device(device)
        self.seed = seed
        self.args = args
        self.args.vae_path = None
        self.pipe = None
        self.generator = None
        self.scheduler = LCMScheduler.from_pretrained(args.pretrained_teacher_model, subfolder="scheduler")
        self.pretrained_teacher_model = args.pretrained_teacher_model


    def clear(self) -> None:
        del self.pipe
        self.pipe = None
        torch.cuda.empty_cache()
        gc.collect()

    def set_pipe_and_generator(self): 
        # disable NSFW filter to avoid black images, **ONLY for the benchmark evaluation** 
        text_encoder = CLIPTextModel.from_pretrained(
                self.pretrained_teacher_model, subfolder="text_encoder"
            ).to(self.device, dtype=torch.float16).requires_grad_(False)
        if self.args.vae_path is not None:
            from diffusers import AutoencoderTiny
            print("loading Tiny VAE")
            vae = AutoencoderTiny.from_pretrained(self.args.vae_path).to(self.device, dtype=torch.float16).requires_grad_(False)
        else:
            if 'Realistic_Vision' in self.pretrained_teacher_model:
                print("loading stabilityai/sd-vae-ft-ema...")
                vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device, dtype=torch.float16).requires_grad_(False)
            else:
                vae = AutoencoderKL.from_pretrained(
                        self.pretrained_teacher_model, subfolder="vae"
                    ).to(self.device, dtype=torch.float16).requires_grad_(False)
        tokenizer = CLIPTokenizer.from_pretrained(
                    self.pretrained_teacher_model, subfolder="tokenizer"
            )
        unets = []
        for path in self.weight_folders:
            if 'prune' in path or 'Prune' in path or 'ours' in path:
                MODEL_OBJ = CustomUNet2DConditionModel
            else:
                MODEL_OBJ = UNet2DConditionModel

            unets.append(
                MODEL_OBJ.from_pretrained(
                    path, subfolder="unet"
                ).to(self.device, dtype=torch.float16).requires_grad_(False)
            )

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers
                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    self.args.logger.log(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                for unet in unets:
                    unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.pipe = HybridLCMPipeline.from_pretrained(
            self.pretrained_teacher_model,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=self.scheduler, # using LCM scheduler
            unet=unets[0]
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unets = unets
        total_step, step_config = self.get_step_config(self.args)
        print(f'total_step={total_step}, step_config={step_config}')
        self.pipe.step_config = step_config
        self.total_step = total_step 
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
    
        
        # skip safety_checker
        self.pipe.safety_checker = None

    def set_lora_ckpt(self):
        if self.args.is_lora_checkpoint:
            self.args.logger.log(" ** use lora checkpoints")
            hybrid_load_and_set_lora_ckpt(
                                pipe=self.pipe,
                                weight_path=[os.path.join(path, 'lora.pt') for path in self.args.lora_weight_path],
                                config_path=[os.path.join(path, 'lora_config.json') for path in self.args.lora_weight_path],
                                dtype=torch.float16)


    def generate(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, save_path=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path
        )
        return out.images
    
    def generate_latents(self, prompt: Union[str, List[str]],  negative_prompt: Union[str, List[str]] = None, img_sz: int = 512,  guidance_scale: float = 7.5, num_images_per_prompt=1, output_type = "latent", save_path=None) -> List[Image.Image]:
        out = self.pipe(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = self.total_step,
            height = img_sz,
            width = img_sz,
            output_type = "latent",
            generator = self.generator,
            guidance_scale = guidance_scale,
            num_images_per_prompt = num_images_per_prompt,
            save_path = save_path
        )
        return out.images
    
    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters())

    def get_step_config(self, args):
        assert len(args.steps) > 0 
        total_step = sum(args.steps)
        assert total_step > 0
        assert len(self.weight_folders) == len(args.steps)
        step_config = {
            "step":{},
            "name":{}
        }
        total_step = 0 
        for index, model_step in enumerate(args.steps):
            for i in range(model_step):
                step_config["step"][total_step] = index
                total_step += 1
        for index, model_name in enumerate(self.weight_folders):
            step_config["name"][index] = model_name.split("/")[-1]

        return total_step, step_config

    def get_sdm_params(self):
        params_str = ""
        for index in range(len(self.pipe.unets)):
            model_name = self.weight_folders[index].split("/")[-1]
            cur_unet = self._count_params(self.pipe.unets[index])
            params_str += f" {model_name}: {(cur_unet/1e6):.1f}M"
        params_text_enc = self._count_params(self.pipe.text_encoder)
        params_image_dec = self._count_params(self.pipe.vae.decoder)
        return_str =  params_str + f"TextEnc {(params_text_enc/1e6):.1f}M; ImageDec {(params_image_dec/1e6):.1f}M)"
        return return_str




def load_and_set_lora_ckpt(pipe, weight_path, config_path, dtype):
    device = pipe.unet.device

    with open(config_path, "r") as f:
        lora_config = json.load(f)
    lora_checkpoint_sd = torch.load(weight_path, map_location=device)
    unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
    text_encoder_lora_ds = {
        k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
    }

    unet_config = LoraConfig(**lora_config["peft_config"])
    pipe.unet = LoraModel(unet_config, pipe.unet)
    set_peft_model_state_dict(pipe.unet, unet_lora_ds)

    if "text_encoder_peft_config" in lora_config:
        text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
        pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
        set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe


def hybrid_load_and_set_lora_ckpt(pipe, weight_paths, config_paths, dtype):
    device = pipe.unet.device

    assert len(config_paths) == len(weight_paths)
    assert len(config_paths) == pipe.unets
    
    for index in len(config_paths):
        config_path = config_paths[index]
        weight_path = weight_paths[index]
        with open(config_path, "r") as f:
            lora_config = json.load(f)
        lora_checkpoint_sd = torch.load(weight_path, map_location=device)
        unet_lora_ds = {k: v for k, v in lora_checkpoint_sd.items() if "text_encoder_" not in k}
        text_encoder_lora_ds = {
            k.replace("text_encoder_", ""): v for k, v in lora_checkpoint_sd.items() if "text_encoder_" in k
        }

        unet_config = LoraConfig(**lora_config["peft_config"])
        pipe.unets[index] = LoraModel(unet_config, pipe.unets[index])
        set_peft_model_state_dict(pipe.unets[index], unet_lora_ds)

        if "text_encoder_peft_config" in lora_config:
            text_encoder_config = LoraConfig(**lora_config["text_encoder_peft_config"])
            pipe.text_encoder = LoraModel(text_encoder_config, pipe.text_encoder)
            set_peft_model_state_dict(pipe.text_encoder, text_encoder_lora_ds)

        if dtype in (torch.float16, torch.bfloat16):
            pipe.unets[index].half()
            pipe.text_encoder.half()

        pipe.to(device)
        return pipe