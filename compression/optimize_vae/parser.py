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
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="sd_models/runwayml--stable-diffusion-v1-5",
        # required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--student_model_name_or_path",
        type=str,
        default=None,
        # required=True,
        help="Path to student model.",
    )
    parser.add_argument(
        "--pretrained_vae_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained VAE model with better numerical stability. More details: https://github.com/huggingface/diffusers/pull/4038.",
    )
    parser.add_argument(
        "--vae_model_class",
        type=str,
        default='normal',
        # required=True,
        help="Model class of vae model, tiny for teasd",
        choices=('tiny', 'normal')
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="Laion_aesthetics_5plus_1024_33M",
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument('--MASTER', type=str,
                        help='MASTER_NODE')      
    parser.add_argument(
        "--type",
        type=str,
        default="base",
        choices=["base", "small", "tiny"],
        help=("Distill type."),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=True,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        default=True,
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--validation_prompts",
        type=str,
        default=None,
        nargs="+",
        help=("A set of prompts evaluated every `--validation_epochs` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-distill-base",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=10000)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=4000000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--prediction_type",
        type=str,
        default='epsilon',
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers."
    )
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=5,
        help="Run validation every X epochs.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Where only evaluate model."
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help="Where to report."
    )
    parser.add_argument(
        "--report_name",
        type=str,
        default="tensorboard",
        help="Where to report."
    )
    parser.add_argument(
        "--log_steps",
        type=int,
        default=100,
        help="Log losses every X steps.",
    )
    ## Loss parameters for training
    parser.add_argument(
        "--disc_start",
        type=int,
        default=50001
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=0.000001
    )
    parser.add_argument(
        "--disc_weight",
        type=float,
        default=0.5
    )
    # parser.add_argument(
    #     "--input_dir",
    #     type=str,
    #     default='evaluation/coco2017/tiny_model_0601'
    # )
    # parser.add_argument(
    #     "--input_root_real",
    #     type=str,
    #     default='evaluation/coco2017/val2017_resize_5'
    # )


    parser.add_argument("--gen_dir", type=str, default = 'evaluation/coco2017/tiny_model_0601')
    parser.add_argument("--input_root_real", type=str,default = 'evaluation/coco2017/val2017_resize_5')
    parser.add_argument("--autoencoderkl_path", type=str,default = 'pretrained_models/runwayml--stable-diffusion-v1-5')
    parser.add_argument("--perceptual_weight", type=float,default = 0.001)
    parser.add_argument("--train_all_steps_start", type=float,default = 100000)
    parser.add_argument("--device_ids", type=str,default = '0, 1, 2, 3')
    parser.add_argument("--experiment_name", type=str,default = 'tinyvae_from_scratch_baseline_512_cgan_0603')
    parser.add_argument("--if_use_average", type=bool,default = False)
    parser.add_argument("--gan_rounds", type=float,default = 500)
    parser.add_argument("--if_penalize", type=float ,default = 0)
    parser.add_argument("--if_saturation_aug", type=bool ,default = False)
    parser.add_argument("--if_use_highfrec", type=bool ,default = False)
    parser.add_argument("--if_maxpool", type=bool ,default = False)
    parser.add_argument("--if_bianyuan", type=bool ,default = False)
    parser.add_argument("--gan_weight", type=float ,default = 1.0)
    parser.add_argument("--penalize_weight", type=float ,default = 1.0)
    parser.add_argument("--replay_buffer", type=bool ,default = False)
    parser.add_argument("--disc_learning_rate", type=float ,default = 0.0001)
    parser.add_argument("--disc_round", type=float ,default = 20)
    parser.add_argument("--disc_step_before_start", type=float ,default = 2000)
    parser.add_argument("--gan_mode_1_threshold", type=float ,default = 60000)
    parser.add_argument("--gan_mode_1to2_disc_step", type=float ,default = 5000)
    parser.add_argument("--gan_2_rounds", type=float ,default = 4)
    parser.add_argument("--disc_2_rounds", type=float ,default = 20)
    parser.add_argument("--if_color_augment", type=bool ,default = False)
    parser.add_argument("--if_diff_loss", type=bool ,default = False)
    parser.add_argument("--model_checkpoint", type=str ,default = 'outputs_ldm/fintune_dino_fuxian/checkpoint-40000/vae.bin')
    parser.add_argument("--if_gan", type=bool ,default = False)
    parser.add_argument("--add_lq_input", type=bool ,default = False)
    parser.add_argument("--if_decoder_distil", type=bool ,default = False)
    parser.add_argument("--if_gan_add_l1", type=bool ,default = False)
    parser.add_argument("--if_use_kl", type=bool ,default = False)
    parser.add_argument("--visual_path", type=str ,default = 'datasets/taesd_visual')
    parser.add_argument("--real_path", type=str ,default = 'datasets/coco2017_resize')

    


    args = parser.parse_args()

 
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
