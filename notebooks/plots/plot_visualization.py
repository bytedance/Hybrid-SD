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



def load_visualization_imgs():
    image_dir = "/mnt/bn/bytenn-yg2/datasets/eval_img"
    transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".jpg")]
    visual_imgs = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        image = transform(image).unsqueeze(0)
        visual_imgs.append(image)
    return torch.cat(visual_imgs, dim=0)

def load_visualization_imgs_1024():
    image_dir = "/mnt/bn/bytenn-yg2/datasets/eval_1024"
    transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
    ])
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]
    visual_imgs = []
    for image_file in image_files:
        image = Image.open(image_file).convert("RGB")
        image = transform(image).unsqueeze(0)
        visual_imgs.append(image)
    return torch.cat(visual_imgs, dim=0)




def plot_visualization(device="cuda:0"):
    with torch.no_grad():
        weight_dtype = torch.float16
        model = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        model.to(weight_dtype)
        
        taesd = AutoencoderTiny.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd").eval().to(weight_dtype)
        
        config_path = "/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/ours_edge_vae/combine_45k_pixelfilter/taesd_config.json"
        tiny_path = "/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/ours_edge_vae/combine_45k_pixelfilter/fintune_dino_combine_pixelfilter-checkpoint-45000-vae.bin"
        tiny_config = AutoencoderTiny.load_config(config_path)
        tiny = AutoencoderTiny.from_config(tiny_config).eval()
        tiny.load_state_dict(torch.load(tiny_path), strict=True)
        tiny = tiny.to(weight_dtype)
        imgs = load_visualization_imgs()
        imgs.to(weight_dtype)
        with torch.cuda.amp.autocast():
            model, taesd, tiny =  model.to(device), taesd.to(device), tiny.to(device)
            imgs = imgs.to(device)
            
            # SD decode
            latent = model.encode(imgs).latent_dist.sample()
            out_imgs = model.decode(latent)['sample']
            save_path = "/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/notebooks/figures/compare_vae_rec"
            os.makedirs(save_path, exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save(os.path.join( save_path,  str(img_idx) + '_SD.png'))
                
            # taesd decode
            latent = taesd.encode(imgs).latents
            out_imgs = taesd.decode(latent)['sample']
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save(os.path.join( save_path,  str(img_idx) + '_taesd.png'))
                
            # taesd decode
            latent = tiny.encode(imgs).latents
            out_imgs = tiny.decode(latent)['sample']
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save(os.path.join( save_path,  str(img_idx) + '_tiny.png'))


def plot_unet_visualization(device="cuda:0"):
    weight_dtype = torch.float16
    from compression.prune_sd.LCM.Scheduling_LCM import LCMScheduler
    from compression.hybrid_sd.diffusers.pipline_hybrid_LCM import StableDiffusionPipeline
    from diffusers import UNet2DConditionModel
    SD14_path = "/mnt/bn/bytenn-yg2/pretrained_models/CompVis--stable-diffusion-v1-4"
    LCM_SD_path = "results/lcm_sd14_2w/checkpoint-20000"
   
    
    unet = UNet2DConditionModel.from_pretrained(
        LCM_SD_path, subfolder="unet",
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        SD14_path,
        unet=unet,
        scheduler=LCMScheduler.from_pretrained(SD14_path, subfolder="scheduler"),
        torch_dtype=weight_dtype,
    )
    pipeline.to(device)
    generator = torch.Generator(device=device).manual_seed(1134)
    with torch.no_grad():
        validation_prompts = [
        "realistic portrait photography of beautiful girl, pale skin, golden earrings, summer golden hour, kodak portra 800, 105 mm fl. 8.",
        "Astronaut in a jungle, detailed, 8k",
        "2045 train station city landscale, conceptart, illustration, highly detailed, artwork, hyper realistic, in style of Ivan aivazovsky",
        "A warm and loving family portrait, highly detailed, hyper-realistic, 8k resolution, photorealistic, soft and natural lighting,",
        "2045 train station city landscale, conceptart, illustration, highly detailed, artwork, hyper realistic, in style of Ivan aivazovsky",
        "a beautiful hyperrealistic vase with a huge bouquet of celestial flowers, still life classical renaissance lighting ar 3/4"
        ]
        latents = []
        for _, prompt in enumerate(validation_prompts):
            autocast_ctx = torch.autocast(device_type="cuda", dtype=weight_dtype)

            with autocast_ctx:
                latent = pipeline(
                    prompt=prompt,
                    num_inference_steps=8,
                    num_images_per_prompt=1,
                    guidance_scale=7,
                    output_type="latent",
                    generator=generator,
                ).images
            latents.append(latent.squeeze())
        latents = torch.stack(latents, dim=0)
        model = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        model.to(weight_dtype)
        
        taesd = AutoencoderTiny.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd").eval().to(weight_dtype)
        
        config_path = "/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/ours_edge_vae/combine_45k_pixelfilter/taesd_config.json"
        tiny_path = "/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/ours_edge_vae/combine_45k_pixelfilter/fintune_dino_combine_pixelfilter-checkpoint-45000-vae.bin"
        tiny_config = AutoencoderTiny.load_config(config_path)
        tiny = AutoencoderTiny.from_config(tiny_config).eval()
        tiny.load_state_dict(torch.load(tiny_path), strict=True)
        tiny = tiny.to(weight_dtype)
       
        latents.to(weight_dtype)
        with torch.cuda.amp.autocast():
            model, taesd, tiny =  model.to(device), taesd.to(device), tiny.to(device)
     
            
            # SD decode
            out_imgs = model.decode(latents/model.config.scaling_factor)['sample']
            save_path = "/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/notebooks/figures/compare_vae_unet"
            os.makedirs(save_path, exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save(os.path.join( save_path,  str(img_idx) + '_SD.png'))
                
            # taesd decode
            out_imgs = taesd.decode(latents)['sample']
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save(os.path.join( save_path,  str(img_idx) + '_taesd.png'))
                
            # taesd decode
            out_imgs = tiny.decode(latents)['sample']
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save(os.path.join( save_path,  str(img_idx) + '_tiny.png'))


def add_box():
    import numpy as np
    from PIL import Image, ImageDraw

    # Load the original image
    original_image = Image.open('/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/notebooks/figures/compare_vae_unet/0_tiny.png')  # Replace with your image path
    original_array = np.array(original_image)

    # Define the bounding box (x1, y1, x2, y2)
    box_size = 100
    #bounding_box = (10, 400, 110, 500) # green
    #bounding_box = (180, 120, 280, 220) # face
    #bounding_box = (200, 200, 300, 300) #flower
    bounding_box = (100, 50, 200, 150) # girl
    x1, y1, x2, y2 = bounding_box

    # Extract the region to be enlarged
    region = original_array[y1:y2, x1:x2]

    # Enlarge the region
    enlarged_region = Image.fromarray(region).resize((region.shape[1] * 2, region.shape[0] * 2), Image.LANCZOS)

    # Create a new blank image with the same shape as the original
    new_width = original_array.shape[1]
    new_height = original_array.shape[0]
    new_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))

    # Paste the original image
    new_image.paste(original_image, (0, 0))

    # Draw a red box around the original bounding box
    draw = ImageDraw.Draw(new_image)
    draw.rectangle(bounding_box, outline='red', width=5)

    # Paste the enlarged region at the specified location (right bottom)
    new_image.paste(enlarged_region, (original_array.shape[1]-box_size*2, original_array.shape[0] - box_size*2))
    
    # Draw a red box around the original bounding box
    new_box = (original_array.shape[1]-box_size*2,original_array.shape[1]-box_size*2,original_array.shape[1],original_array.shape[1] )
    draw = ImageDraw.Draw(new_image)
    draw.rectangle(new_box, outline='red', width=5)


    # Save or show the new image
    new_image.save('/mnt/bn/bytenn-yg2/liuhj/hybrid_sd/bytenn_diffusion_tools/notebooks/figures/compare_vae_unet/0_tiny_enlarge.png') # Display the image


if __name__ == "__main__":
    add_box()
    #plot_visualization()
    #plot_unet_visualization()