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

import argparse
from glob import glob
import logging
import math
import os
import random
import shutil
from pathlib import Path
import copy

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.optim import RAdam
from torch.utils.data import DataLoader
import torchvision.transforms.functional  as TF
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed #, DeepSpeedPlugin
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.utils import ContextManagers

import diffusers
from diffusers import  StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available



from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
from compression.optimize_vae.models.autoencoder_tiny import AutoencoderTiny, AutoencoderTinyWS
from compression.optimize_vae.dino import TinyVaeDino
from compression.prune_sd.calflops import calculate_flops

from compression.optimize_vae.webdata_laion import WebDataset
from compression.optimize_vae.parser import parse_args
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.aggregation import MeanMetric
from torchvision.io import read_image
from cleanfid import fid
from torch.utils.data import DataLoader, Dataset
from compression.optimize_vae.data_utils import *
from taming.modules.losses.vqperceptual import * 
logger = get_logger(__name__, log_level="INFO")


def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images


def train():
    args = parse_args()

    # os.environ['MASTER_ADDR']=args.MASTER
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # deepspeed_plugin = DeepSpeedPlugin(
    #     zero_stage=1,
    # )
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" if args.report_to =="wandb" else "tensorboard",
        project_config=accelerator_project_config,
        split_batches=True,
        #deepspeed_plugin=deepspeed_plugin,  #use deepspeed
    )
    
    # if args.report_to == "wandb":
    #     from diffusers.utils import is_wandb_available
    #     if not is_wandb_available():
    #         raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    #     import wandb

    if args.report_to == 'wandb':
        import wandb
        accelerator.init_trackers(
        project_name="optimize_vae",
        config=args,
        init_kwargs={"wandb":{"name":args.experiment_name}},
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        # datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
  
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        #text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        teacher = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        vae = AutoencoderTinyWS.from_pretrained(args.student_model_name_or_path)
        #teacher = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet") 
        
        
    ema_vae = copy.deepcopy(vae)
    # count_params(vae, verbose=True)
    flops, macs, params = calculate_flops(
        model=vae, 
        input_shape=(1, 3, 512, 512),
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
        print_results=False)
    logger.info(f'#Params={params}, FLOPs={flops}, MACs={macs}')
    
    teacher.requires_grad_(False)
    vae.requires_grad_(True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            teacher.enable_xformers_memory_efficient_attention()
            print("using xformers")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            # ema_vae.save_pretrained(os.path.join(output_dir, "vae_ema"))
            
            for i, model in enumerate(models):
                if isinstance(model, AutoencoderTinyWS):
                    model.save_pretrained(os.path.join(output_dir, f"vae"))
                else:
                    accelerator.save(model.state_dict(), os.path.join(output_dir, f"discriminator.ckpt"))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            # ema_vae.load_state_dict(os.path.join(input_dir, "vae_ema"))

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = AutoencoderTinyWS.from_pretrained(input_dir, subfolder="vae")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()

    # refer to configs/autoencoder/autoencoder_kl_64x64x3.yaml 
    # disc_start: 50001
    # kl_weight: 0.000001
    # disc_weight: 0.5
    # modified to

    
    Tinyvae_with_Dnet = TinyVaeDino(disc_start=args.disc_start)
    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_D = optimizer_cls(
        Tinyvae_with_Dnet.discriminator.parameters(),
        lr=args.learning_rate*50,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_e = optimizer_cls(
        vae.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    train_dataloader = WebDataset(args.train_data_dir, tokenizer_path=args.pretrained_model_name_or_path, batch_size=args.train_batch_size, size=args.resolution)
    use_colorjitter = True

    def color_augment(im): #[0,1] images
        scale = 0.5 + torch.randn(3, 3).to(im.device).to(im.dtype)
        blend = torch.rand(3, 1, 1).to(im.device).to(im.dtype)
        return (scale @ im.flatten(2)).view(im.shape).clamp(0, 1) * blend + (1 - blend) * im
    # eval coco fid dataset
    
    def transform_func(examples):
        val_transform = transforms.Compose([transforms.ToTensor()])
        examples['image'] = [val_transform(img.convert("RGB")) for img in examples['image']]
        return examples

    real_path = args.real_path
    if accelerator.is_main_process:
        from datasets import load_dataset
        from torchvision import transforms
        img_list = [os.path.join(real_path, name) for name in os.listdir(real_path) if name.endswith(".png")]
        eval_dataset = load_dataset('imagefolder', data_files=img_list,split='train')
        eval_dataset = eval_dataset.with_transform(transform_func)['image']
        eval_batch = 50 
        eval_loader = DataLoader(eval_dataset, batch_size=eval_batch, num_workers=0, shuffle=False)


    # Scheduler and math around the number of training steps.
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    lr_scheduler_D = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_D,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler_e = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_e,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )


    # Prepare everything with our `accelerator`.
    vae, Tinyvae_with_Dnet, optimizer, optimizer_D,optimizer_e, train_dataloader, lr_scheduler, lr_scheduler_D, lr_scheduler_e  = accelerator.prepare(vae, Tinyvae_with_Dnet, optimizer, optimizer_D, optimizer_e, train_dataloader, lr_scheduler, lr_scheduler_D,lr_scheduler_e)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision


    teacher.to(weight_dtype).to(accelerator.device)
    vae.to(weight_dtype)
    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps

            num_update_steps_per_epoch = 40000000 #forced
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    train_loss = 0.0

    def _momentum_update_key_encoder(encoder_q,encoder_k):
        """
        Momentum update of the key encoder. 
        k: ema model,  q: current model
        """
        m=0.99
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 -m)

    #### pixel filter
    if args.add_lq_input:
        round_input_filter=0
    else:
        round_input_filter = 1

    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            teacher.requires_grad_(False)
            teacher.eval()
            vae.requires_grad_(True)
            ema_vae.to(vae.device)
            ema_vae.requires_grad_(False)

            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    #progress_bar.update(1)
                    pass
                continue
            with accelerator.accumulate(vae, Tinyvae_with_Dnet):
                k = 6
                if global_step % (k+1) ==0:
                    optimizer_idx = 0
                elif global_step % (k+1) ==1:
                    optimizer_idx = 2
                else:
                    optimizer_idx = 1
                
                if global_step < args.disc_start: # only train discriminator
                    optimizer_idx = 1


                if optimizer_idx == 0: # decoder
                    accelerator.unwrap_model(vae).encoder.eval()
                    accelerator.unwrap_model(vae).decoder.train()
                    accelerator.unwrap_model(Tinyvae_with_Dnet).discriminator.eval()
                elif optimizer_idx == 2: # encoder
                    accelerator.unwrap_model(vae).encoder.train()
                    accelerator.unwrap_model(vae).decoder.eval()
                    accelerator.unwrap_model(Tinyvae_with_Dnet).discriminator.eval()
                    
                else: # discriminator phase
                    vae.eval()
                    accelerator.unwrap_model(Tinyvae_with_Dnet).discriminator.train()
                

                input = batch["pixel_values"] # shape [b, 3, h, w]
                if random.random() > 0.9:
                    input = color_augment(input.add(1.).div(2.))
                    input = input.mul(2.).sub(1.)


                if round_input_filter % 3==0 and optimizer_idx == 0:
                    with torch.autocast(device_type="cuda", dtype=weight_dtype):
                        accelerator.unwrap_model(vae).eval() 
                        with torch.no_grad():
                            t_latent = teacher.encode(input).latent_dist.sample()
                            lq_latent = accelerator.unwrap_model(vae).encode(input.add(1.).div(2.)).latents
                            lq_output =  accelerator.unwrap_model(vae).decode(lq_latent)['sample']
                            s_output = teacher.decode(t_latent)['sample']

                            input,s_output = input.add(1.0).div(2.),s_output.add(1.0).div(2.) ### [0,1]
                            l1_loss_lq = torch.abs(lq_output.contiguous() - s_output.contiguous())
                            one = torch.ones_like(l1_loss_lq)
                            threshold = l1_loss_lq.flatten().quantile(q=0.5).item() #### 50%的元素进行训练
                            l1_loss_lq = torch.where(l1_loss_lq < threshold, one, l1_loss_lq)
                            input = torch.where(l1_loss_lq>=1.0, one, input)
                            input = input.mul(2.0).sub(1.0) ### [-1,1]


                if optimizer_idx==0 and args.if_decoder_distil:
                    with torch.autocast(device_type="cuda", dtype=weight_dtype):
                        with torch.no_grad():
                            t_latent = teacher.encode(input).latent_dist.sample()  # teacher input [-1, 1] latent       
                            s_output = teacher.decode(t_latent)['sample']
                            t_latent = t_latent * teacher.config.scaling_factor
                            noise_index = 0.25 * torch.rand(1).cuda() #[0,0.25] latent degradation
                            lq_latent = t_latent  * (1 - noise_index )
                            

                        lq_output = accelerator.unwrap_model(vae).decode(lq_latent)['sample'] # student output             
                        last_layer = accelerator.unwrap_model(vae).decoder.layers[-1].weight
                        lq_output = lq_output.mul(2.).sub(1.)  # convert input, s_output [0,1] -> [-1,1]
                        dec_loss, log = Tinyvae_with_Dnet(s_output, lq_output, optimizer_idx = optimizer_idx, global_step=global_step, last_layer = last_layer)
                    

                if optimizer_idx==0 and (not args.if_decoder_distil):
                    #### decoder latent
                    with torch.autocast(device_type="cuda", dtype=weight_dtype):
                        with torch.no_grad():
                            t_latent = teacher.encode(input).latent_dist.sample() * teacher.config.scaling_factor  # teacher input [-1, 1] latent       
                            noise_index = 0.25 * torch.rand(1).cuda() #[0,0.25] 
                            lq_latent = t_latent  * (1 - noise_index )

                        lq_output = accelerator.unwrap_model(vae).decode(lq_latent)['sample'] # student output             
                        last_layer = accelerator.unwrap_model(vae).decoder.layers[-1].weight
                        lq_output = lq_output.mul(2.).sub(1.)  # convert input, s_output [0,1] -> [-1,1]
                        dec_loss, log = Tinyvae_with_Dnet(input, lq_output, optimizer_idx = optimizer_idx, global_step=global_step, last_layer = last_layer)
                    


                if optimizer_idx==2:
                    #### encoder 
                    with torch.autocast(device_type="cuda", dtype=weight_dtype):
                        with torch.no_grad():
                            t_latent = teacher.encode(input).latent_dist.sample() * teacher.config.scaling_factor  # teacher input [-1, 1] latent       
                        with torch.autocast(device_type="cuda", dtype=weight_dtype):
                            lq_latent = accelerator.unwrap_model(vae).encode(input.add(1.).div(2.)).latents
                            latent_loss = torch.abs(lq_latent.contiguous() - t_latent).contiguous()
                            dec_loss = torch.sum(latent_loss) / latent_loss.shape[0]
                            log = {"encoder_l1":dec_loss.clone().detach().mean()}


                if optimizer_idx==1:
                    #### discriminator 
                    with torch.autocast(device_type="cuda", dtype=weight_dtype):
                        with torch.no_grad():
                            t_latent = teacher.encode(input).latent_dist.sample() 
                            if args.if_decoder_distil:
                                s_output = teacher.decode(t_latent)['sample']
                            
                            t_latent = t_latent * teacher.config.scaling_factor  # teacher input [-1, 1] latent       
                            noise_index = 0.25 * torch.rand(1).cuda() #[0,0.25] 
                            lq_latent = t_latent  * (1 - noise_index )
                            
                        lq_output = accelerator.unwrap_model(vae).decode(lq_latent)['sample'] # student output             
                        lq_output = lq_output.mul(2.).sub(1.)  # convert input, s_output [0,1] -> [-1,1]
                        if args.if_decoder_distil:
                            dec_loss, log = Tinyvae_with_Dnet(s_output, lq_output, optimizer_idx = optimizer_idx, global_step=global_step, last_layer = None)
                        else:
                            dec_loss, log = Tinyvae_with_Dnet(input, lq_output, optimizer_idx = optimizer_idx, global_step=global_step, last_layer = None)




                loss  = dec_loss
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                if optimizer_idx == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                elif optimizer_idx == 2:
                    optimizer_e.step()
                    optimizer_e.zero_grad()
                else:
                    optimizer_D.step()
                    optimizer_D.zero_grad()
                lr_scheduler.step()
                lr_scheduler_D.step()
                lr_scheduler_e.step()
            # Checks if the accelerator has performed an optimization step behind the scenes
            if optimizer_idx == 0  and accelerator.is_main_process: # log generator loss
                loss_type = "gen_loss"
                logger.info(f"global step: {global_step}, {loss_type}: {avg_loss.detach().item()}")
                accelerator.log({
                    loss_type: loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]
                }, step=global_step)

                accelerator.log(
                log
                , step=global_step)

            if optimizer_idx == 1 and accelerator.is_main_process: # log discriminator loss
                loss_type = "dis_loss"
                logger.info(f"global step: {global_step}, {loss_type}: {avg_loss.detach().item()}")
                accelerator.log(
                    log
                    , step=global_step)  

            if optimizer_idx == 2 and accelerator.is_main_process: # log discriminator loss
                loss_type = "encoder_loss"
                logger.info(f"global step: {global_step}, {loss_type}: {avg_loss.detach().item()}")
                accelerator.log(
                    log
                    , step=global_step)        

            # logging training info to dir
            if accelerator.sync_gradients:
                global_step += 1
                if optimizer_idx == 0 and args.add_lq_input:
                    round_input_filter +=1

                if accelerator.is_main_process:
                    with torch.no_grad():
                        _momentum_update_key_encoder(accelerator.unwrap_model(vae), ema_vae)  # updata EMA
                if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                    # # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    os.makedirs(os.path.join(args.output_dir,f"checkpoint-{global_step}"), exist_ok=True)
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", "vae.bin")
                    accelerator.save(accelerator.unwrap_model(vae).state_dict(), save_path)
                    ema_path = os.path.join(args.output_dir, f"checkpoint-{global_step}", "ema_vae.bin")
                    accelerator.save(ema_vae.state_dict(), ema_path)
                    if global_step > args.disc_start:
                        dis_save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}","disc.bin")
                        accelerator.save(accelerator.unwrap_model(Tinyvae_with_Dnet).discriminator.state_dict(), dis_save_path)
                    logger.info(f"Saved state to {save_path}")

                accelerator.free_memory()

            """
            Evaluate and generate inter image samples 
            """
            if accelerator.is_main_process and (global_step % (args.checkpointing_steps/5)==0 or global_step in [1,2, 10, 50, 80, 100,300,500,1200]):
                vae.eval()
                teacher.eval()
                with torch.no_grad():
                    input = load_visualization_imgs().to(accelerator.device).to(weight_dtype)# [-1, 1]
                    base = load_visualization_imgs(image_dir=args.visual_path).to(accelerator.device).to(weight_dtype).add(1).div(2) # [-1, 1]
                    with torch.autocast(device_type="cuda"):
                        t_latent = teacher.encode(input).latent_dist.sample() # teacher latent                        
                        s_latent = accelerator.unwrap_model(vae).encode(input.add(1.).div(2.)).latents # student latent
                        t_output  = teacher.decode(t_latent)['sample'] # teacher output
                        s_output = accelerator.unwrap_model(vae).decode(s_latent)['sample'] # student output
                                                

                    diff = torch.cat([base, s_output, 4.*(base-s_output).abs()],dim=-1)
                    if args.report_to == "wandb":
                        accelerator.log( { "base_vs_finetuned_images":[
                                                        wandb.Image(
                                                            (( diff[img_idx].detach() +0) * 255.).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                            caption="student_generated_img")
                                                            for img_idx in range(diff.shape[0]) 
                                                        ] }, step = global_step)
                        if global_step < 10:
                            accelerator.log( {  "gt_images":[
                                                            wandb.Image(
                                                                (( input[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                                    caption="gt_img")
                                                                for img_idx in range(input.shape[0]) 
                                                            ] }, step = global_step) 
                            accelerator.log( {  "teacher_generated_images":[
                                                            wandb.Image(
                                                                (( t_output[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                                    caption="teacher_generated_img")
                                                                for img_idx in range(t_output.shape[0]) 
                                                            ] }, step = global_step)
                        accelerator.log( { "student_generated_images":[
                                                        wandb.Image(
                                                            (( s_output[img_idx].detach() +0) * 255.).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                            caption="student_generated_img")
                                                            for img_idx in range(s_output.shape[0]) 
                                                        ] }, step = global_step)
                        if global_step < 10:
                            accelerator.log( { "teacher_latent_images":[
                                                            wandb.Image(
                                                                (( ((t_latent[img_idx].detach()[0:3] * teacher.config.scaling_factor).div(2*3).add(0.5) )) * 255).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                                    caption="teacher_latent")
                                                                for img_idx in range(t_latent.detach().shape[0]) 
                                                            ] }, step = global_step)
                            accelerator.log( { "student_latent_images":[
                                                            wandb.Image(
                                                                (( (s_latent[img_idx].detach()[0:3]).div(2*3).add(0.5)) * 255).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                                    caption="student_latent")
                                                                for img_idx in range(s_latent.detach().shape[0]) 
                                                            ] }, step = global_step)
    
                vae.train()
            if accelerator.sync_gradients:
                # if accelerator.is_main_process and ((global_step+1) % 10000==0):
                if accelerator.is_main_process and ((global_step+1) % 100==0):
                # if accelerator.is_main_process and (global_step ==1):
                    # start eval and calculate fid metrics
                    accelerator.unwrap_model(vae).eval()
                    torch.cuda.empty_cache()
                    with torch.no_grad(): #[0,1] input decoder
                        out_path = os.path.join(args.output_dir, "coco2017", "coco2017_" +  str(global_step))
                        os.makedirs(out_path, exist_ok=True)
                        for batch_id, data in enumerate(eval_loader):
                            data = data.to(vae.device, dtype=weight_dtype)
                            latent =  accelerator.unwrap_model(vae).encode(data).latents
                            output = accelerator.unwrap_model(vae).decode(latent)['sample'].mul(255).clamp(0, 255).cpu().byte() 
                            end = batch_id*eval_batch + eval_batch if (batch_id*eval_batch+eval_batch)<=len(img_list) else len(img_list)
                            data_files = img_list[batch_id*eval_batch:end]
                            for file_id in range(len(output)):
                                save_path_final = os.path.join(out_path, data_files[file_id].split('/')[-1])
                                img = output[file_id]
                                TF.to_pil_image(img).convert('RGB').save(save_path_final)

                    
                    logger.info('##### evaluate')
                    psnr_score, lpips_score , fid_score = evaluation_coco2017(vae, real_path,out_path,weight_dtype,logger)
                    accelerator.log({"psnr":psnr_score}, step=global_step)
                    accelerator.log({"lpips":lpips_score}, step=global_step)
                    accelerator.log({"FID":fid_score}, step=global_step)
                    accelerator.free_memory()
                    # vae.train()
                    accelerator.unwrap_model(vae).train()

            logs = {
                "step_loss": train_loss/(global_step+1), 
                "lr": lr_scheduler.get_last_lr()[0]
            }

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()



if __name__ == "__main__":
    train()
