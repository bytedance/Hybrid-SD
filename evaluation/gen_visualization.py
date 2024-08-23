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
    #transforms.Normalize([0.5], [0.5])
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




def gen_visual_sd(device): 
    with torch.no_grad():
        weight_dtype = torch.float16
        model = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        model.to(weight_dtype)
        imgs = load_visualization_imgs_1024()
        imgs.to(weight_dtype)
        with torch.cuda.amp.autocast():
            model =   model.to(device)
            imgs = imgs.to(device)
            latent = model.encode(imgs).latent_dist.sample()
            out_imgs = model.decode(latent)['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/' + str(img_idx) + '.png')

    
def gen_visual_tiny(device): 
    with torch.no_grad():
        weight_dtype = torch.float32
        #model = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        model = AutoencoderTiny.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd").eval()
        model.to(weight_dtype)
        imgs = load_visualization_imgs()
        imgs.to(weight_dtype)
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            model =   model.to(device)
            imgs = imgs.to(device)
            latent = model.encode(imgs).latents
            out_imgs = model.decode(latent)['sample']
            from torchvision.transforms import  GaussianBlur
            gaussian =   GaussianBlur((3,3), sigma=(0.1, 1.))
            out_imgs = gaussian(out_imgs)
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_gs', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_gs/' + str(img_idx) + '.jpg')

def gen_visual_tiny_1024(device): 
    with torch.no_grad():
        weight_dtype = torch.float32
        #model = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        model = AutoencoderTiny.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd").eval()
        model.to(weight_dtype)
        imgs = load_visualization_imgs_1024()
        imgs.to(weight_dtype)
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            model =   model.to(device)
            imgs = imgs.to(device)
            latent = model.encode(imgs).latents
            out_imgs = model.decode(latent)['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_gs', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_gs/' + str(img_idx) + '.jpg')

def gen_visual_tiny_distilled(): 
    with torch.no_grad():
        weight_dtype = torch.float32
        #model = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        model_config = AutoencoderTiny.load_config("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd")
        model = AutoencoderTiny.from_config(model_config).eval()
        model.load_state_dict(torch.load("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/distill_tiny_vae_dist/checkpoint-56000/vae.bin"), strict=True)
        model.to(weight_dtype)
        imgs = load_visualization_imgs()
        imgs.to(weight_dtype)
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            model =   model.cuda()
            imgs = imgs.cuda()
            latent = model.encode(imgs).latents
            out_imgs = model.decode(latent)['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_dis', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_dis/' + str(img_idx) + '.jpg')

def gen_visual_tiny_finetune(): 
    # input [-1, 1] output [0,1]
    with torch.no_grad():
        weight_dtype = torch.float32
        model_config = AutoencoderTinyWS.load_config("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd")
        model = AutoencoderTiny.from_config(model_config).eval()
        model.load_state_dict(torch.load("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/finetune_tinyvae/checkpoint-40000/vae.bin"), strict=True)
        model.to(weight_dtype)
        imgs = load_visualization_imgs() # [0,1]
        imgs.mul(2).sub(1)
        imgs.to(weight_dtype)
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            model =   model.cuda()
            imgs = imgs.cuda()
            latent = model.encode(imgs).latents
            out_imgs = model.decode(latent)['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_finetune', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() +1.) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tiny_finetune/' + str(img_idx) + '.jpg')



def gen_visual_lite(): 
    with torch.no_grad():
        weight_dtype = torch.float32
        from compression.optimize_vae.models.litevae_kl import LiteVAE
        #model = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        vae_config = LiteVAE.load_config("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/litevae_dino_train_from_scratch_baseline_512")
        model = LiteVAE.from_config(vae_config).eval()
        model.load_state_dict(torch.load("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/litevae_dino_train_from_scratch_baseline_512/checkpoint-198000/vae.bin"), strict=True)
        model.to(weight_dtype)
        imgs = load_visualization_imgs()
        imgs.to(weight_dtype)
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            model =   model.cuda()
            imgs = imgs.cuda()
            latent = model.encode(imgs).latent_dist 
            out_imgs = model.decode(latent.sample())['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/litevae', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() + 1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/litevae/' + str(img_idx) + '.jpg')



def cross_visual_tiny_sd():
    with torch.no_grad():
        weight_dtype = torch.float32
        SD = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        tiny = AutoencoderTiny.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd")
        tiny.to(weight_dtype), SD.to(weight_dtype)
        imgs = load_visualization_imgs() #[0,1]
        imgs.to(weight_dtype)
        tiny.eval()
        SD.eval()
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            SD, tiny =  SD.cuda(), tiny.cuda()
            imgs = imgs.cuda()
            #latent = SD.encode(imgs).latent_dist.sample() * SD.config.scaling_factor
            latent = SD.encode(imgs.mul(2.).sub(1.)).latent_dist.sample() * SD.config.scaling_factor
           
            #latent = tiny.encode(imgs.mul(2.).sub(1.)).latents
            out_imgs = tiny.decode(latent)['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/SD_en_tiny_de01', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() +1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/SD_en_tiny_de01/' + str(img_idx) + '.jpg')


def cross_visual_tinyws_sd():
    with torch.no_grad():
        weight_dtype = torch.float32
        SD = AutoencoderKL.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5", subfolder="vae").eval()
        tiny = AutoencoderTinyWS.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd")
        tiny.to(weight_dtype), SD.to(weight_dtype)
        imgs = load_visualization_imgs() #[0,1]
        imgs.to(weight_dtype)
        tiny.eval()
        SD.eval()
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            SD, tiny =  SD.cuda(), tiny.cuda()
            imgs = imgs.cuda()
            latent = SD.encode(imgs.mul(2.).sub(1.)).latent_dist.sample() * SD.config.scaling_factor #[-1,1]
            #latent = SD.encode(imgs).latent_dist.sample() * SD.config.scaling_factor #[0,1]

            
            #latent = tiny.encode(imgs).latents # [0,1]
            out_imgs = tiny.decode(latent)['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/SD_en11_tiny_de11', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                print(out_imgs[img_idx].min(), out_imgs[img_idx].max())
                img = ((out_imgs[img_idx].detach() +1) * 127.5).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                #img = ((out_imgs[img_idx].detach() +0) * 255.).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/SD_en11_tiny_de11/' + str(img_idx) + '.jpg')


def gen_visual_tinyws_finetuned():
    with torch.no_grad():
        weight_dtype = torch.float32
        tiny = AutoencoderTinyWS.from_pretrained("/mnt/bn/bytenn-yg2/pretrained_models/madebyollin--taesd")
        tiny.load_state_dict(torch.load("/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/outputs/finetune_tinyvae_dino/checkpoint-80000/ema_vae.bin"), strict=True)
        tiny.to(weight_dtype)
        imgs = load_visualization_imgs() #[0,1]
        imgs.to(weight_dtype)
        tiny.eval()
        with torch.cuda.amp.autocast(dtype=weight_dtype):
            tiny =  tiny.cuda()
            imgs = imgs.cuda()
            latent = tiny.encode(imgs).latents #[0,1] -> [0,1]
            out_imgs = tiny.decode(latent)['sample']
            os.makedirs('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tinyws_ema', exist_ok=True)
            for img_idx in range(out_imgs.shape[0]):
                img = ((out_imgs[img_idx].detach() +0) * 255.).clamp(0, 255).permute(1,2,0).to(torch.uint8).cpu().numpy()
                img = Image.fromarray(img)
                img.save('/mnt/bn/bytenn-yg2/liuhj/bytenn_diffusion_tools/evaluation/visual_img/tinyws_ema/' + str(img_idx) + '.jpg')


if __name__ == "__main__":
    gen_visual_tinyws_finetuned()
    #cross_visual_tinyws_sd()
    #device = torch.device("cuda:2")
    #gen_visual_tiny_1024(device)