import torch
from diffusers import DiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from torch import Generator
import os 

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True


def inference(model_path, prompt, negative_prompt, output_path):
    with torch.inference_mode():
        gen = Generator("cuda")
        gen.manual_seed(1674753452)
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False)
        pipe.to('cuda')
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.unet.to(device='cuda', dtype=torch.float16, memory_format=torch.channels_last)

        for i in range(3):
            img = pipe(prompt=prompt,negative_prompt=negative_prompt, width=512, height=512, num_inference_steps=25, guidance_scale = 7, num_images_per_prompt=1, generator = gen).images[0]
            img.save(f"{output_path}/{i}.png")


if __name__=="__main__":
    model_path = 'segmind/small-sd' #model card
    # model_path = 'segmind/tiny-sd' #model card
    # model_path = 'SG161222/Realistic_Vision_V4.0' #origin model card
    prompt = "Faceshot Portrait of pretty young (18-year-old) Caucasian wearing a high neck sweater, (masterpiece, extremely detailed skin, photorealistic, heavy shadow, dramatic and cinematic lighting, key light, fill light), sharp focus, BREAK epicrealism"
    negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    output_path = os.path.join("output", model_path.split("/")[-1])
    os.makedirs(output_path, exist_ok=True)

    inference(model_path, prompt, negative_prompt, output_path)

