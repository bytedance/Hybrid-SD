import PIL.Image as Image
from pathlib import Path
from tqdm import tqdm, trange
from diffusers.utils import make_image_grid
from typing import List


def generate_images(pipe, prompts, image_dir, num_images_per_prompt=2, save=True, n_rows=1, n_cols=2):
    if isinstance(prompts, str) and Path(prompts).is_file():
        with open(prompts, 'r') as f:
            prompts = f.readlines()

    prompts = [prompt.strip() for prompt in prompts]
    images_list = []

    for i in trange(len(prompts), desc="Sampling"):
        prompt = prompts[i]
        print(f"generate img for prompt: {prompt}")
        images = pipe(prompt, num_images_per_prompt=num_images_per_prompt).images
        images_list.append(images)

        if save:
            grid_image = make_image_grid(images, rows=n_rows, cols=n_cols)

            # grid_image = image_grid(images, rows=n_rows, cols=n_cols)
            imgname = str(i) + '_' + '_'.join(prompt.split(' ')[-4:]) + '.png'
            imgpath = f'{image_dir}/{imgname}'
            grid_image.save(imgpath)
            print(f"save image to {imgpath}")
        
    return images_list, prompts
