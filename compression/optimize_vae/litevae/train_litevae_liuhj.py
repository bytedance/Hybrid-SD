
#!/usr/bin/env python
# coding=utf-8
""" 
VAE optimization 
"""
import argparse
from glob import glob
import logging
import math
import os
import sys
sys.path.append("/mnt/bn/bytenn-data2/liuhj/pylib")
sys.path.insert(0, "/mnt/bn/bytenn-data2/liuhj/bytenn_diffusion_tools/compression/optimize_vae")
from utils import count_params
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
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed #, DeepSpeedPlugin
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import sys
sys.path.append('/mnt/data/group/liuhj/pylib')
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from contperceptual import LPIPSWithDiscriminator


from webdata_hj import WebDataset
from parser import parse_args

MASTER_PORT = os.environ.get('MASTER_PORT')
MASTER_ADDR = os.environ.get('MASTER_ADDR')
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
RANK = os.environ.get('RANK')
INT_RANK = int(RANK)
LOCAL_RANK = os.environ.get('LOCAL_RANK', 0)
INT_LOCAL_RANK = int(LOCAL_RANK)
print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, RANK:{}, LOCAL_RANK:{}".format(MASTER_ADDR, MASTER_PORT, WORLD_SIZE,
                                                                                   RANK, LOCAL_RANK))
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
    if args.report_to == "wandb":
        from diffusers.utils import is_wandb_available
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb
        wandb.init(project="optimize_vae")
        wandb.config.update(args)

    os.environ['MASTER_ADDR']=args.MASTER
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

    if args.report_to == 'wandb':
        accelerator.init_trackers(
        project_name="optimize_vae", 
        config=args
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
        # datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)
        print("随机种子已固定为:", args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")

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
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        #teacher = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet") 
        
        #TODO: not used for VAE
        #unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", ignore_mismatched_sizes=True)  
    


    count_params(vae, verbose=True)

    # Freeze unet and text_encoder
    vae.requires_grad_(True)
    #text_encoder.requires_grad_(False)
    #unet.requires_grad_(False)
    #teacher.requires_grad_(False)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            #unet.enable_xformers_memory_efficient_attention()
            print("using xformers")
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "student_cd"))
                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
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
    disc_start = 50001
    LPIPS_with_discriminator = LPIPSWithDiscriminator(disc_start=disc_start, kl_weight=1e-6, disc_weight=0.5)

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
        LPIPS_with_discriminator.discriminator.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    #train_dataset = coco2014(args.train_data_dir)

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(texts, is_train=True):
        inputs = tokenizer(texts, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def collate_fn(examples):
        images = [image['image'].convert("RGB") for image in examples]
        texts = [image['text'] for image in examples]
        img_pixel_values = [train_transforms(image) for image in images]
        input_ids = tokenize_captions(texts)

        pixel_values = torch.stack(img_pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values, "input_ids": input_ids}



    url = "/mnt/bn/bytenn-data2/liuhj/Laion_aesthetics_5plus_1024_33M"
    train_dataloader = WebDataset(url, batch_size=args.train_batch_size, size = args.resolution)

    # # DataLoaders creation:
    # train_dataloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     shuffle=False,
    #     collate_fn=collate_fn,
    #     batch_size=args.train_batch_size,
    #     num_workers=args.dataloader_num_workers,
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    #     overrode_max_train_steps = True
    args.max_train_steps = 4000000
    args.num_train_epochs = 10000
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        #num_cycles=5000,
    )

    # Prepare everything with our `accelerator`.


    optimizer, optimizer_D, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, optimizer_D, train_dataloader, lr_scheduler)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision

    # Move text_encode and vae to gpu and cast to weight_dtype
    #unet.to(accelerator.device, dtype=weight_dtype)
    #text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    #teacher.to(accelerator.device, dtype=weight_dtype)
    LPIPS_with_discriminator.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    #num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    # if overrode_max_train_steps:
    #     args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
   
   
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    #logger.info(f"  Num examples = {len(train_dataset)}")
    #logger.info(f"  Num Epochs = {args.num_train_epochs}")
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

    # Only show the progress bar once on each machine.

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(accelerator.device)
    #uc = text_encoder( tokenize_captions([""]*args.train_batch_size ).to(accelerator.device))[0]    #unconditional inputs
    def append_dims(x, target_dims):
        """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
        dims_to_append = target_dims - x.ndim
        if dims_to_append < 0:
            raise ValueError(
                f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
            )
        return x[(...,) + (None,) * dims_to_append]
    # def denoise(model, x_t, t, c, w):
    #     dims = x_t.ndim
    #     c_skip = append_dims(1/torch.sqrt( alphas_cumprod[t]), dims).to(accelerator.device)
    #     c_out = append_dims(-torch.sqrt((1- alphas_cumprod[t] )/ alphas_cumprod[t]), dims).to(accelerator.device)
    #     c_in = append_dims(torch.ones(x_t.shape[0],), dims).to(accelerator.device)
    #     model_output = model(c_in * x_t, t, w, c).sample
    #     denoised = c_out * model_output + c_skip * x_t
    #     return model_output, denoised

   

    @torch.no_grad()
    def ddim_solver(teacher, noisy_latents, latents, c, uc, t, next_t, w, alphas_cumprod):
        """
        DDIM update k=20 steps
        samples: noisy_latents
        x0: latents
        """
        e=teacher(noisy_latents,t,c).sample
        ue = teacher(noisy_latents,t,uc).sample
        dims=latents.ndim
        alpha,alpha1=append_dims(alphas_cumprod[t].sqrt(),dims),append_dims(alphas_cumprod[next_t].sqrt(),dims)
        sigma,sigma1=append_dims((1-alphas_cumprod[t]).sqrt(),dims),append_dims((1-alphas_cumprod[next_t]).sqrt(),dims)
        w1=append_dims(w, dims)

        phi=alpha1/alpha*noisy_latents-sigma1*(sigma*alpha1/alpha/sigma1-1)*e-noisy_latents
        uphi=alpha1/alpha*noisy_latents-sigma1*(sigma*alpha1/alpha/sigma1-1)*ue-noisy_latents
        noisy_latents1=noisy_latents+(1+w1)*phi-w1*uphi
        return noisy_latents1


    def Pseudo_Huber_loss(x, y, weight):
        c = torch.tensor(0.03)
        weight  = torch.tensor(weight)
        return weight * torch.mean( (torch.sqrt((x - y)**2 + c**2) -c), dim=(1,2,3))


    def wasserstein_loss(u1, s1, u2, s2):
        """
        quantifies how much mass need to be moved from a distribution to a distribution
        can be explicit under gaussion distribution
        w^2 = |u1 - u2|^2 + (s1+s2 - 2(\sqrt{c2}*c1*\sqrt{c2})^0.5)
        """
        w_loss = ((u1-u2)**2) + torch.exp(0.5*s1) + torch.exp(0.5*s2) - 2.* torch.sqrt((torch.exp(0.5*s1)*(torch.exp(0.5*s2))))
        #replace with softplus
        #w_loss = ((u1-u2)**2) + solftplus(0.5*s1) + solftplus(0.5*s2) - 2.* torch.sqrt((solftplus(0.5*s1)*(solftplus(0.5*s2))))
        return w_loss.mean()

    train_loss = 0.0

    vae_factor =  0.18215 # vae.config.scaling_factor for SD 1.5

    def _momentum_update_key_encoder(encoder_q,encoder_k):
        """
        Momentum update of the key encoder. 
        k: ema model,  q: current model
        """
        m=0.999
        for param_q, param_k in zip(encoder_q.parameters(), encoder_k.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 -m)

    ema_vae  = copy.deepcopy(vae) # ema_model
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            #unet.requires_grad_(False)
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    #progress_bar.update(1)
                    pass
                continue
            if global_step > disc_start and (global_step%2)==0:
                optimizer_idx = 1 # train discriminator
            else:
                optimizer_idx = 0 # train generator

            if optimizer_idx == 0:
                #generator phase 
                LPIPS_with_discriminator.discriminator.eval()
                vae.train()
                optimizer.zero_grad()
            else:
                # discriminator phase
                LPIPS_with_discriminator.discriminator.train()
                vae.eval()
                optimizer_D.zero_grad()


            input = batch["pixel_values"].to(weight_dtype) # shape [b, 3, h, w]

            # posteriors : class DiagonalGaussianDistribution(),  distribution,
            # to get a sample, use posteriors.sample()
            posteriors = vae.encode(input).latent_dist #.sample()
                    
            dec = vae.decode(posteriors.sample())['sample'] # shape [b, 3, h, w]
            
            # support last_layer weight for adaptive loss weight
            last_layer = vae.decoder.conv_out.weight
            

            # optimizer_idx
            # posteriors = posteriors.kl()
            loss, log = LPIPS_with_discriminator(input, dec, posteriors, optimizer_idx = optimizer_idx , global_step=global_step, last_layer=last_layer)
            if accelerator.is_main_process and global_step % 100 == 0:
                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        if optimizer_idx == 0:
                            tracker.writer.add_histogram(tag="gen_loss/task", values=loss.detach().view(-1), global_step=global_step)
                        else:
                            tracker.writer.add_histogram(tag="dis_loss/task", values=loss.detach().view(-1), global_step=global_step)
                    elif args.report_to == "wandb":
                            pass

        
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
            train_loss += avg_loss.item() / args.gradient_accumulation_steps
            # Backpropagate
            loss.backward()
            #accelerator.backward(loss)
            # if accelerator.sync_gradients:
            #     accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
            if optimizer_idx==0:
                optimizer.step()
            else:
                optimizer_D.step()
            lr_scheduler.step()
            _momentum_update_key_encoder(vae,ema_vae)
            # Checks if the accelerator has performed an optimization step behind the scenes

            # logging training info to dir
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.log({
                    "train_loss": loss.detach().item(),
                }, step=global_step)

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
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

                        os.makedirs(args.output_dir+f"/checkpointscm",exist_ok=True)
                        save_path = args.output_dir+f"/checkpointscm/vae_{global_step}.bin"
                        accelerator.save(accelerator.unwrap_model(vae).state_dict(),save_path)
                        accelerator.save(ema_vae.state_dict(), args.output_dir+f"/checkpointscm/ema_{global_step}.bin")
                        logger.info(f"Saved state to {save_path}")
              
                accelerator.free_memory()
            if accelerator.is_main_process and global_step % 10== 0:
                print("global step:", global_step, " loss: ",avg_loss.detach().item())
         
            """
            Evaluate and generate inter image samples 
            """


            # TODO
            if accelerator.is_main_process and (global_step % (args.checkpointing_steps/10)==0 or global_step in [1,2,3,4, 100,300,500,1200]):
                # eval on selected dataset
                # vae.eval()
                # with torch.no_grad():
                    # load imgs
                if args.report_to == "wandb":
                    accelerator.log( {  "gt_images":[
                                                    wandb.Image(
                                                        (( input[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                            caption="gt_img")
                                                        for img_idx in range(input.shape[0]) 
                                                    ] }, step=global_step)
                    accelerator.log( { "vae_generated_images":[
                                                    wandb.Image(
                                                        (( dec[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy(),
                                                        caption="vae_generated_img")
                                                        for img_idx in range(dec.shape[0]) 
                                                    ] }, step=global_step)
            
            # add evaluation code
            # tmp_save_path = args.output_dir+f"/inter_samples"
            # os.makedirs(tmp_save_path, exist_ok=True)
            # if global_step % (args.checkpointing_steps/10)==0 or global_step in [1,100,300,500,1200]:
            #     # eval on selected dataset
            #     student.eval()
                
            #     with torch.no_grad():
            #         #eval(model=student, iters=global_step, tmp_save_path=tmp_save_path)
            #         prompts, names, local_index = eval_dataset()
                   
            #         batch_size = len(names)
            #         num_process=1
            #         from accelerate import PartialState
            #         distributed_state = PartialState()
            #         device = distributed_state.device
            #         stepsn = noise_scheduler.config.num_train_timesteps
            #         length = len(local_index) // batch_size
            #         for i in range(length):
            #             batch = prompts[i*batch_size*num_process : (i+1)*batch_size*num_process]
            #             nns=names[i*batch_size*num_process : (i+1)*batch_size*num_process]
            #             # partial_state will split batch to each gpu
            #             with distributed_state.split_between_processes(local_index) as local_indexs:
            #                 bsz = len(local_indexs)
            #                 nn = []
            #                 prompt = []
            #                 for i in range(bsz):
            #                     prompt.append(batch[local_indexs[i]])
            #                     nn.append(nns[local_indexs[i]])
            #                 #id0 = i*batch_size*num_process + bsz*distributed_state.local_process_index
            #                 #noisy_latents = noise_scheduler.add_noise(latents, torch.randn_like(latents), torch.tensor([stepsn-1], device=ts.device) )
            #                 seed_everything(42)
            #                 noisy_latents = torch.randn(bsz,4,64,64).to(device)  #用原照片xt生成的图片会更好
            
            #                 encoder_hidden_states = text_encoder(tokenize_captions(prompt ).to(device))[0]
            #                 # sampling
            #                 # Sample a random timestep for each image
            #                 eval_step = 6
            #                 steps_list = [4, 6]
            #                 #CM
            #                 ts = torch.linspace(stepsn, 0, eval_step, device=device,dtype=int)
            #                 ts[0] -= 1
            #                 w=torch.Tensor([8.0]).cuda()
            #                 w=w.repeat(bsz)
            #                 x, _, _ = denoise_sde(student, noisy_latents, torch.tensor([ts[0]], device=ts.device), encoder_hidden_states, w, sample=False)
            #                 for j in range(1,len(ts)):
            #                     # same noise every iteration
            #                     x = noise_scheduler.add_noise(x, torch.randn_like(x), torch.tensor([ts[j]], device=ts.device))
            #                     x, _, _ = denoise_sde(student, x, ts[j:j+1].repeat(bsz),encoder_hidden_states, w, sample=False)
            #                     cur_steps = j+ 1 # current steps
            #                     if cur_steps in steps_list:
            #                         img = vae.decode(x/vae.config.scaling_factor, return_dict=False)[0]
            #                         if img.shape[0]==1:
            #                             img = img.permute(0,2,3,1).detach()
            #                         else:
            #                             img = img.squeeze(0).permute(0,2,3,1).detach()
            #                         img = ((img + 1) * 127.5).clamp(0, 255).to(torch.uint8).cpu().numpy()
            #                         for i0 in range(len(nn)):
            #                             name=nn[i0]
            #                             name=name.split('/')[-1][:-4]
            #                             title = prompt[i0].replace("\\"," ").replace("/"," ").replace(".","")
            #                             im = Image.fromarray(img[i0])
            #                             tmppath = tmp_save_path
            #                             if not os.path.exists(tmppath):
            #                                 os.makedirs(tmppath)
            #                             saveimgpath = os.path.join(tmppath, str(global_step) + '__' + str(cur_steps) +'_' + str(name) + '.jpg')
            #                             im.save(saveimgpath)#
            #             student.train()

            logs = {
                "step_loss": train_loss/(global_step+1), 
                "lr": lr_scheduler.get_last_lr()[0]
            }
            #progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    '''
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(student)

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet
        )
        pipeline.save_pretrained(args.output_dir)
    '''
    accelerator.end_training()




if __name__ == "__main__":
    train()
