import torch
import diffusers
from pathlib import Path
import os
from PIL import Image
import copy
from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
from compression.optimize_vae.models.autoencoder_tiny import AutoencoderTinyWS
from diffusers.models.autoencoders.autoencoder_tiny import AutoencoderTiny
from torchvision import transforms
from diffusers import DiffusionPipeline



def gen_visual_txt2img(device, vae_dtype="SD", weight_dtype = torch.float32, path=None):
    with torch.no_grad():
        pipe = DiffusionPipeline.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", torch_dtype = weight_dtype)
        pipe = pipe.to(device)
        prompt = ["a high-resolution image or illustration of a diverse group of people facing me, each displaying a distinct range of emotions. The image should be in 8K resolution or provide the highest quality available.",
                  "A tennis player trying to save the ball", 
                  "A group portrait of several people, highly detailed, hyper-realistic, 8k resolution, photorealistic, cinematic lighting, studio lighting",
                  "A diverse group of friends, highly detailed, hyper-realistic, 8k resolution, photorealistic, vibrant and varied lighting, creative and engaging composition, genuine and joyful expressions, a wide range of ages, ethnicities, ",
                  "A warm and loving family portrait, highly detailed, hyper-realistic, 8k resolution, photorealistic, soft and natural lighting,",
                  "Colorful hot air balloon festival, hyper-saturated balloons against deep blue sky",
                  "Tropical rainforest, saturated greens, exotic plants, hanging vines, and mossy trees, rivers",
                  "Futuristic skyscraper, intricate grid facade, glass and steel structure, modern architecture",
                  "a cute bird with colorful and saturated feathers, close scene, green grass, hyper-resolution",
                  ]
        generator = [torch.Generator(device=device).manual_seed(40) for i in range(len(prompt))]
        latents = pipe(prompt, generator=generator, output_type="latent").images
        if vae_dtype == "SD":
            vae = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
            vae = vae.to(device)
            latents = latents / pipe.vae.config.scaling_factor 
        elif vae_dtype == "tiny":
            vae = AutoencoderTiny.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd").eval()
            vae = vae.to(device)
        elif vae_dtype == "tiny_finetune":
            vae = AutoencoderTinyWS.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd").eval()
            vae.load_state_dict(torch.load(path), strict=True)
            vae = vae.to(device)
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            out_imgs = vae.decode(latents)['sample']
            name = vae_dtype
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_unet_img/' + name, exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                #img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = out_imgs[img_idx].detach().mul(255.).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_unet_img/' + name + '/' + str(img_idx) + '.png')



if __name__ == "__main__":
    device = torch.device("cuda:6")
    path = "/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/finetune_tinyvae_dino_degrade_latent2/scal_0_4/checkpoint-6000/vae.bin"
    #path = "/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/finetune_tinyvae_dino/checkpoint-80000/vae.bin"
    vae_dtype = "tiny_finetune" # "tiny" #"SD"  #"tiny" # "tiny"  # "tiny"  # 
    gen_visual_txt2img(device, vae_dtype=vae_dtype, path=path)
