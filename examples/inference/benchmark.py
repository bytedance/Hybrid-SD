import os
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from pytorch_lightning import seed_everything
import argparse
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image

def warmup(pipe, prompt, tgate=False):
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(2):
        if tgate:
            _, = pipe.tgate(prompt, output_type='pt', inference_num_per_image=50).images
        else:
            _ = pipe(prompt, output_type='pt').images

def clear_cache(pipe):
    del pipe
    torch.cuda.empty_cache()
    
def benchmark_baseline(model, prompt, n_steps, infer_times):
    logging.info("Running baseline...")
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")
    warmup(pipe, prompt)
    start_time = time.time()
    for _ in range(infer_times):
        ori_output = pipe(prompt, num_inference_steps=n_steps).images
    latency = (time.time() - start_time) / infer_times
    logging.info(f"Baseline: {latency:.2f} seconds")
    clear_cache(pipe)
    return ori_output[0]

def benchmark_deepcache(model, prompt, n_steps, infer_times, cache_interval=5):
    from DeepCache.sd.pipeline_stable_diffusion import StableDiffusionPipeline as DeepCacheStableDiffusionPipeline

    logging.info("Running DeepCache...")
    pipe = DeepCacheStableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")
    warmup(pipe, prompt)
    start_time = time.time()
    for _ in range(infer_times):
        deepcache_output = pipe(
            prompt, 
            num_inference_steps=n_steps,
            cache_interval=cache_interval, cache_layer_id=0, cache_block_id=0,
            uniform=False, pow=1.4, center=15,
            output_type='pt'
        ).images
    latency = (time.time() - start_time) / infer_times
    logging.info(f"DeepCache(I={cache_interval}){latency:.2f} seconds")
    clear_cache(pipe)
    return deepcache_output[0]

def benchmark_tome(model, prompt, n_steps, infer_times):
    import tomesd
    logging.info("Running TOME...")
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")
    tomesd.apply_patch(pipe, ratio=0.5)
    warmup(pipe, prompt)
    start_time = time.time()
    for _ in range(infer_times):
        tome_image = pipe(prompt, num_inference_steps=n_steps).images
    latency = (time.time() - start_time) / infer_times
    logging.info(f"TOME: {latency:.2f} seconds")
    clear_cache(pipe)
    return tome_image[0]
    
def benchmark_tgate(model, prompt, n_steps, infer_times):
    from tgate import TgateSDLoader
    logging.info("Running TGATE...")
    pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")
    gate_step = 8
    pipe = TgateSDLoader(
        pipe,
        gate_step=gate_step,
        num_inference_steps=n_steps
    ).to("cuda")
    warmup(pipe, prompt, tgate=True)
    start_time = time.time()
    for _ in range(infer_times):
        tagate_image = pipe.tgate(
            prompt,
            gate_step=gate_step,
            num_inference_steps=n_steps
        ).images    
    latency = (time.time() - start_time) / infer_times
    logging.info(f"T-GATE: {latency:.2f} seconds")
    clear_cache(pipe)
    return tagate_image[0]
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='/mnt/bn/ycq-lq/hf_models/Byte_SD1.5_V1')
    parser.add_argument("--prompt", type=str, default='a photo of an astronaut on a moon')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--infer_times", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    args = parser.parse_args()
    
    seed = args.seed
    seed_everything(args.seed)

    prompt = args.prompt

    ori_output = benchmark_baseline(args.model, args.prompt, args.num_inference_steps, args.infer_times)
    deepcache_output = benchmark_deepcache(args.model, args.prompt,  args.num_inference_steps, args.infer_times, cache_interval=5)
    deepcache_output = benchmark_deepcache(args.model, args.prompt,  args.num_inference_steps, args.infer_times, cache_interval=3)
    tome_output = benchmark_tome(args.model, args.prompt,  args.num_inference_steps, args.infer_times)
    tgate_output = benchmark_tgate(args.model, args.prompt, args.num_inference_steps, args.infer_times)

    # outputs = [ori_output, deepcache_output, tome_output]
    # array_images = np.stack([np.array(img) for img in outputs])
    # merged_image = Image.fromarray(array_images)
    # merged_image.save('outputs/benchmark_output.png')