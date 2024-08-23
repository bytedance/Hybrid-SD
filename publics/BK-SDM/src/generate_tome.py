# ------------------------------------------------------------------------------------
# Copyright 2023–2024 Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

import os
import argparse
import time
from utils.inference_pipeline_tome import InferencePipeline
from utils.misc import get_file_list_from_csv, change_img_size

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/mnt/bn/ycq-yg/hf_models/Byte_SD1.5_V1")    
    parser.add_argument("--save_dir", type=str, default="./results/bk-sdm-small",
                        help="$save_dir/{im256, im512} are created for saving 256x256 and 512x512 images")
    parser.add_argument("--unet_path", type=str, default=None)   
    parser.add_argument("--data_list", type=str, default="/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_30k/metadata.csv")    
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img_sz", type=int, default=512)
    parser.add_argument("--img_resz", type=int, default=256)
    parser.add_argument("--batch_sz", type=int, default=1)
    
    # ToME SD arguments
    parser.add_argument("--tome", action='store_true')
    parser.add_argument("--tome_ratio", type=float, default=0.5)
    parser.add_argument("--tome_sx", type=int, default=2)
    parser.add_argument("--tome_sy", type=int, default=2)
    parser.add_argument("--tome_max_downsample", type=int, default=1)
    parser.add_argument("--tome_merge_corssattn", action='store_true')
    parser.add_argument("--tome_merge_mlp", action='store_true')

    # TGATE arguments
    parser.add_argument("--tgate", action='store_true')
    parser.add_argument("--tgate_step", type=int, default=8)

    # DeepCache arguments
    parser.add_argument("--deepcache", action='store_true')
    parser.add_argument("--deepcache_cache_internal", type=int, default=5)
    parser.add_argument("--deepcache_cache_layer_id", type=int, default=0)
    parser.add_argument("--deepcache_cache_block_id", type=int, default=0)

    args = parser.parse_args()
    return args

def check_images(image_names, save_dir_src):
    new_image_names = []
    for img_name in image_names:
        img_path = os.path.join(save_dir_src, img_name)
        if os.path.exists(img_path):
            continue
        new_image_names.append(img_name)
    return new_image_names

if __name__ == "__main__":
    args = parse_args()

    pipeline = InferencePipeline(weight_folder = args.model_id,
                                seed = args.seed,
                                device = args.device,
                                tome_args = {
                                    'tome': args.tome,
                                    'ratio': args.tome_ratio,
                                    'sx': args.tome_sx,
                                    'sy': args.tome_sy,
                                    'max_downsample': args.tome_max_downsample,
                                    'merge_crossattn': args.tome_merge_corssattn,
                                    'merge_mlp': args.tome_merge_mlp
                                },
                                deepcache_args = {
                                    'deepcache': args.deepcache,
                                    'cache_internal': args.deepcache_cache_internal,
                                    'cache_layer_id': args.deepcache_cache_layer_id,
                                    'cache_block_id': args.deepcache_cache_block_id
                                },
                                tgate_args = {
                                    'tgate': args.tgate,
                                    'tgate_step': args.tgate_step
                                })
                                
    pipeline.set_pipe_and_generator()    

    if args.unet_path is not None: # use a separate trained unet for generation        
        from diffusers import UNet2DConditionModel 
        unet = UNet2DConditionModel.from_pretrained(args.unet_path, subfolder='unet')
        pipeline.pipe.unet = unet.half().to(args.device)
        print(f"** load unet from {args.unet_path}")        

    save_dir_src = os.path.join(args.save_dir, f'im{args.img_sz}') # for model's raw output images
    os.makedirs(save_dir_src, exist_ok=True)
    save_dir_tgt = os.path.join(args.save_dir, f'im{args.img_resz}') # for resized images for ms-coco benchmark
    os.makedirs(save_dir_tgt, exist_ok=True)       

    file_list = get_file_list_from_csv(args.data_list)
    params_str = pipeline.get_sdm_params()
    
    t0 = time.perf_counter()
    for batch_start in range(0, len(file_list), args.batch_sz):
        batch_end = batch_start + args.batch_sz
        
        img_names = [file_info[0] for file_info in file_list[batch_start: batch_end]]
        img_names = check_images(img_names, save_dir_src)

        if len(img_names) == 0:
            continue

        val_prompts = [file_info[1] for file_info in file_list[batch_start: batch_end]]
                    
        imgs = pipeline.generate(prompt = val_prompts,
                                 n_steps = args.num_inference_steps,
                                 img_sz = args.img_sz)

        for i, (img, img_name, val_prompt) in enumerate(zip(imgs, img_names, val_prompts)):
            img.save(os.path.join(save_dir_src, img_name))
            img.close()
            print(f"{batch_start + i}/{len(file_list)} | {img_name} {val_prompt}")
        print(f"---{params_str}")

    pipeline.clear()
    
    change_img_size(save_dir_src, save_dir_tgt, args.img_resz)
    print(f"{(time.perf_counter()-t0):.2f} sec elapsed")
