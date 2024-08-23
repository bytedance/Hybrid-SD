import pdb
import torch
import torch.utils.data
from cleanfid import fid
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from PIL import Image
from tqdm import tqdm
import os
import pdb
import random
import shutil
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2
from torchvision import utils as vutils
from torchvision.io import read_image
from datasets import load_dataset
import argparse
import time
import gc
from compression.prune_sd.calflops.flops_counter import calculate_flops
import torchvision.transforms.functional as TF
# export http_proxy=http://bj-rd-proxy.byted.org:3128  https_proxy=http://bj-rd-proxy.byted.org:3128 no_proxy=code.byted.org
# huggingface-cli login
# python3 -m pip install clean-fid,torchmetrics



def readDir(dirPath):
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = sorted([file for file in os.listdir(dirPath)])
        for f in fileList:
            f = dirPath+'/'+f
            if os.path.isdir(f):
                subFiles = readDir(f)
                allFiles = subFiles + allFiles
            else:
                allFiles.append(f)
        return allFiles
    else:
        return 'Error,not a dir'


def generate_real_image_folder():
    file_to_copy = readDir('./coco2017/val2017')
    file_to_copy = random.sample(file_to_copy,10)
    destination_directory = './coco2017/val2017_real_10'
    if os.path.isdir(destination_directory):
        shutil.rmtree(destination_directory)
    os.makedirs(destination_directory)
    for i in file_to_copy:
        shutil.copy(i,destination_directory)


def transform(examples):
    val_transform = transforms.Compose([transforms.ToTensor()])
    examples['image'] = [val_transform(img.convert("RGB")) for img in examples['image']]
    return examples



def transform_256(examples):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(256),
        transforms.Resize(256,interpolation=transforms.InterpolationMode.BILINEAR)])
    examples['image'] = [val_transform(img.convert("RGB")) for img in examples['image']]
    return examples


def resize(input_dir='./coco2017/val2017', output_dir = './coco2017/val2017_gen'):
    img_list = readDir(input_dir)
    if len(img_list) != 5000:
        raise ValueError(f"the number of images {len(img_list)} is something wrong; 5000 is expected")    
    if os.path.exists(output_dir):       
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((512,512)),
        transforms.Resize((512,512),interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToPILImage()
    ])
    for i in img_list:
        save_path = os.path.join(output_dir,i.split('/')[-1].split('.')[0]+'.png')
        img = Image.open(i)
        img1 = preprocess(img.convert("RGB"))  ##torch.float32, torch.Size([3, 512, 512])
        img1.save(save_path)



class MultiImageDataset(Dataset):
    def __init__(self, root0, root1, is_gt=False):
        super().__init__()
        self.root0 = root0
        self.root1 = root1
        file_names0 = os.listdir(root0)
        file_names1 = os.listdir(root1)

        self.image_names0 = sorted([name for name in file_names0 if name.endswith(".png") or name.endswith(".jpg")])
        self.image_names1 = sorted([name for name in file_names1 if name.endswith(".png") or name.endswith(".jpg")])
        self.is_gt = is_gt
        assert len(self.image_names0) == len(self.image_names1)

    def __len__(self):
        return len(self.image_names0)

    def __getitem__(self, idx):
        img0 = read_image(os.path.join(self.root0, self.image_names0[idx]))
        img1 = read_image(os.path.join(self.root1, self.image_names1[idx]))
        # img0 = np.array(Image.open(os.path.join(self.root0, self.image_names0[idx])).convert("RGB"))
        # img1 = np.array(Image.open(os.path.join(self.root1, self.image_names1[idx])).convert("RGB"))
        batch_list = [img0, img1]
        return batch_list


# from diffusers.models import AutoencoderKL
# model = AutoencoderKL.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder="vae",torch_dtype=torch.float16).eval() ### 用不了
# model = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', subfolder="vae",torch_dtype=torch.float16).eval() ### 用不了

def inference_gen_dir_gpu(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids
    if args.if_fp16:
        weight_dtype=torch.float16
    else:
        weight_dtype=torch.float32

    if args.if_baseline:
        ####### sd1.5 fp16, 没有黑图
        weight_dtype = torch.float16
        from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
        model = AutoencoderKL.from_pretrained('/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5', subfolder="vae",torch_dtype=weight_dtype).eval()
        print(20*'#' + 'loading model : this is AutoencoderKL {}'.format(str(weight_dtype)))

        ####### sd1.5 fp32, 没有黑图
        # weight_dtype = torch.float32
        # from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
        # model = AutoencoderKL.from_pretrained('/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5', subfolder="vae",torch_dtype=weight_dtype).eval()
        # print(20*'#' + 'loading model : this is AutoencoderKL {}'.format(str(weight_dtype)))


        ####### sdxl fp32, 没有黑图
        # weight_dtype = torch.float32
        # from diffusers import DiffusionPipeline
        # base = DiffusionPipeline.from_pretrained(
        #     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=weight_dtype, variant="fp16", use_safetensors=True)
        # model = base.vae.eval()
        # print(20*'#' + 'loading model : this is AutoencoderKL XL {}'.format(str(weight_dtype)))


        ####### sdxl fp16
        # from diffusers import DiffusionPipeline, AutoencoderKL
        # weight_dtype = torch.float16
        # model = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=weight_dtype).eval()
        # print(20*'#' + 'loading model : this is AutoencoderKL XL refine {}'.format(str(weight_dtype)))


        ####### taesd fp32
        # from diffusers import AutoencoderTiny
        # model = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=weight_dtype).eval()  #to(torch.float16)
        # print(20*'#' + 'loading model : this is AutoencoderTiny {}'.format(str(weight_dtype)))
    elif args.model_name == "litevae":
        from compression.optimize_vae.models.litevae_kl import  LiteVAE
        vae_config = LiteVAE.load_config(os.path.join(args.pretrained_model_name_or_path, "config.json"))
        model = LiteVAE.from_config(vae_config).eval()
        model.load_state_dict(torch.load(os.path.join(args.pretrained_model_name_or_path, args.ckpt_name)), strict=True)
    elif args.model_name == "tinyvae":
        from compression.optimize_vae.models.autoencoder_tiny import AutoencoderTiny
        tinyvae_config = AutoencoderTiny.load_config(os.path.join(args.pretrained_model_name_or_path, "config.json"))
        model = AutoencoderTiny.from_config(tinyvae_config).eval()
        model.load_state_dict(torch.load(os.path.join(args.pretrained_model_name_or_path, args.ckpt_name)), strict=True)
    else:
        from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
        model = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").eval()
        print(20*'#' + 'loading model : this is pruner Autoencoder {}'.format(str(weight_dtype)))

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id_list = [i for i in range(len(args.device_ids.split(',')))]
    batch_size = len(device_id_list)
    model = torch.nn.DataParallel(model,device_ids=device_id_list).to(device,dtype=weight_dtype) #dtype=weight_dtype
    #img_list = sorted([os.path.join(args.input_root_real,name) for name in os.listdir(args.input_root_real) if name.endswith(".png")])
    img_list = [os.path.join(args.input_root_real,name) for name in os.listdir(args.input_root_real) if name.endswith(".png")]

    if not os.path.isdir(args.input_dir):
        os.makedirs(args.input_dir)

    if os.path.isdir(args.input_root_gen):
        shutil.rmtree(args.input_root_gen)
    os.makedirs(args.input_root_gen)

    # if os.path.isdir(args.input_root_encoder):
    #     shutil.rmtree(args.input_root_encoder)
    # os.makedirs(args.input_root_encoder)

    # if os.path.isdir(args.input_root_rec):
    #     shutil.rmtree(args.input_root_rec)
    # os.makedirs(args.input_root_rec)

    a = time.time()
    transform_func = transform
    with torch.inference_mode():
        with torch.cuda.amp.autocast(): 
            test_dataset = load_dataset('imagefolder', data_files=img_list,split='train')
            test_dataset = test_dataset.with_transform(transform_func)['image'] ## 变成tensor
            test_loader = DataLoader(test_dataset,batch_size=batch_size,num_workers=args.num_workers,shuffle=False)
            print('total batch:',len(test_dataset)/batch_size)
            for batch_id, data in enumerate(test_loader):
                if batch_id % 100==0:
                    print('Now is batch:',batch_id)
            
                # laji = data.type(torch.HalfTensor).to(device,dtype = torch.HalfTensor)
                # model.module.half()(laji)
                data = data.to(device,dtype=weight_dtype)
                #res_decoder = (model.module(data)['sample'].add(1)).mul(127.5).clamp(0, 255).cpu().byte()  # lhj tiny from scratch style
                res_decoder = model.module(data)['sample'].mul(255).clamp(0, 255).cpu().byte()  #original
                # res_encoder = model.module.encode(data).latent_dist.sample()
                # res_decoder = model.module.decode(res_encoder).sample.mul(255).clamp(0, 255).cpu().byte()
                # res_encoder_to_img = res_encoder.mul_(255).clamp(0, 255).cpu().byte()
                end = batch_id*batch_size+batch_size if (batch_id*batch_size+batch_size)<=len(img_list) else len(img_list)
                data_files = img_list[batch_id*batch_size:end]
                for i in range(len(data_files)):
                    ### encoder的img
                    # save_path_encoder= os.path.join(args.input_root_encoder,data_files[i].split('/')[-1])
                    # img = res_encoder_to_img[i]
                    # TF.to_pil_image(img).convert('RGB').save(save_path_encoder)

                    ### decoder的img
                    save_path_final = os.path.join(args.input_root_gen,data_files[i].split('/')[-1])
                    img = res_decoder[i]
                    TF.to_pil_image(img).convert('RGB').save(save_path_final)

                    ### decode resize成encoder输出的img
                    # save_path_decoder_resize = os.path.join(args.input_root_rec,data_files[i].split('/')[-1])
                    # resize_shape = res_encoder.shape[-1]
                    # decoder_trans_img = transforms.Resize((resize_shape,resize_shape),interpolation=transforms.InterpolationMode.BILINEAR)(res_decoder[i])
                    # TF.to_pil_image(decoder_trans_img).convert('RGB').save(save_path_decoder_resize)
                    # save_path = os.path.join('/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/sdxl/val2017_resize_256',data_files[i].split('/')[-1])
                    # img = data[i]
                    # TF.to_pil_image(img).convert('RGB').save(save_path)



    print('inference成图片耗时:',time.time()-a)
    flops, macs, params = calculate_flops(model=model.module, input_shape=(1, 3, 512, 512),output_as_string=True,output_precision=4,print_detailed=False,print_results=False)## decimal place
    print('模型flops:',flops)
    print('模型macs:',macs)
    print('模型params:',params)
    flops, macs, params = calculate_flops(model=model.module.encoder, input_shape=(1, 3, 512, 512),output_as_string=True,output_precision=4,print_detailed=False,print_results=False)## decimal place
    print('encode模型flops:',flops)
    print('encode模型macs:',macs)
    print('encode模型params:',params)
    flops, macs, params = calculate_flops(model=model.module.decoder, input_shape=(1, 4, 64,64),output_as_string=True,output_precision=4,print_detailed=False,print_results=False)## decimal place
    print('decode模型flops:',flops)
    print('decode模型macs:',macs)
    print('decode模型params:',params)



def main(input_root_real,input_root_gen, args):
    # import lpips
    weight_dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    psnr = PeakSignalNoiseRatio().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device) #normalize=True，net_type='vgg'out of memeory
    ssim = StructuralSimilarityIndexMeasure().to(device)
    inception = InceptionScore().to(device)
    # lpips_model = lpips.LPIPS(net="vgg")

    a = time.time()
    psnr_metric = MeanMetric()
    lpips_metric = MeanMetric()
    ssim_metric = MeanMetric()
    is_metric = MeanMetric()
    distance = []
    dataset = MultiImageDataset(input_root_real, input_root_gen)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # make a json file
    progress_bar = tqdm(dataloader)
    preprocess = transforms.Compose([transforms.ToTensor()])
    with torch.inference_mode():
        for i, batch in enumerate(progress_bar):
            # to cuda
            batch = [img.to(device) for img in batch]
            batch_size = batch[0].shape[0]
            psnr_metric.update(psnr(batch[0].to(weight_dtype), batch[1].to(weight_dtype)).item(), batch_size)
            ssim_metric.update(ssim(batch[0].to(weight_dtype), batch[1].to(weight_dtype)).item(), batch_size)
            # for laji in range(batch[0].shape[0]):
            #     distance.append(lpips_model(preprocess(batch[0][laji].cpu().numpy()),preprocess(batch[1][laji].cpu().numpy())))
            lpips_metric.update(lpips(batch[0]/255, batch[1]/255).item(), batch_size)
            is_metric.update(inception(batch[1]), batch_size)

    fid_score = fid.compute_fid(input_root_real, input_root_gen)
    print('计算指标耗时:',time.time()-a)
    print("PSNR:", psnr_metric.compute().item())
    print("LPIPS:", lpips_metric.compute().item())
    print("ssim:", ssim_metric.compute().item())
    print("IS:", is_metric.compute().item())
    print("FID:", fid_score)
    return {
        "PSNR:", psnr_metric.compute().item(),
        "LPIPS:", lpips_metric.compute().item(),
        "ssim:", ssim_metric.compute().item(),
        "IS:", is_metric.compute().item(),
        "FID:", fid_score,
    }


def eval_metric(input_root_real,input_root_gen, args):
    # import lpips
    weight_dtype = torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    psnr = PeakSignalNoiseRatio().to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device) #normalize=True，net_type='vgg'out of memeory
    ssim = StructuralSimilarityIndexMeasure().to(device)
    inception = InceptionScore().to(device)
    # lpips_model = lpips.LPIPS(net="vgg")

    a = time.time()
    psnr_metric = MeanMetric()
    lpips_metric = MeanMetric()
    ssim_metric = MeanMetric()
    is_metric = MeanMetric()
    distance = []
    dataset = MultiImageDataset(input_root_real, input_root_gen)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # make a json file
    progress_bar = tqdm(dataloader)
    preprocess = transforms.Compose([transforms.ToTensor()])
    with torch.inference_mode():
        for i, batch in enumerate(progress_bar):
            # to cuda
            batch = [img.to(device) for img in batch]
            batch_size = batch[0].shape[0]
            psnr_metric.update(psnr(batch[0].to(weight_dtype), batch[1].to(weight_dtype)).item(), batch_size)
            ssim_metric.update(ssim(batch[0].to(weight_dtype), batch[1].to(weight_dtype)).item(), batch_size)
            # for laji in range(batch[0].shape[0]):
            #     distance.append(lpips_model(preprocess(batch[0][laji].cpu().numpy()),preprocess(batch[1][laji].cpu().numpy())))
            lpips_metric.update(lpips(batch[0]/255, batch[1]/255).item(), batch_size)
            is_metric.update(inception(batch[1]), batch_size)

    fid_score = fid.compute_fid(input_root_real, input_root_gen, num_workers=args.num_workers, use_dataparallel=False)
    print('计算指标耗时:',time.time()-a)
    print("PSNR:", psnr_metric.compute().item())
    print("LPIPS:", lpips_metric.compute().item())
    print("ssim:", ssim_metric.compute().item())
    print("IS:", is_metric.compute().item())
    print("FID:", fid_score)
    return {
        "PSNR:", psnr_metric.compute().item(),
        "LPIPS:", lpips_metric.compute().item(),
        "ssim:", ssim_metric.compute().item(),
        "IS:", is_metric.compute().item(),
        "FID:", fid_score,
    }

def plot_grid(args):
    original_fig = 'evaluation/fig/ori_fig'
    if os.path.isdir(original_fig):
        shutil.rmtree(original_fig)
    os.makedirs(original_fig)

    import matplotlib.pyplot as plt
    path = ['000000000872.png','000000000785.png','000000001296.png','000000506933.png','000000558073.png','000000000285.png']
    wenzi_path = ['000000002592.png','000000004795.png','000000005037.png','000000005477.png','000000006771.png','000000006954.png','000000008211.png','000000011197.png',
    '000000011615.png','000000011760.png','000000015440.png','000000016439.png']
    path = path+wenzi_path
    plt.clf()
    rows = 3
    columns = len(path)//rows
    fig,ax = plt.subplots(rows,columns,figsize=(20, 10))
    for i in range(rows):
        for j in range(columns):
            raw_path = os.path.join(args.input_root_real,path[i*columns + j])
            shutil.copy(raw_path, original_fig) 
            # if i==0: ##原图
            #     img_path = os.path.join(args.input_root_real,path[j])
            #     ax[i][j].set_title(path[j].split('.')[0]+'_real')
            # if i==1: ##生成图
            #     img_path = os.path.join(args.input_root_gen,path[j])
            #     ax[i][j].set_title(path[j].split('.')[0]+'_gen')
            img_path = os.path.join(args.input_root_gen,path[i*columns + j])
            ax[i][j].set_title(path[i*columns + j].split('.')[0])
            ax[i][j].imshow(Image.open(img_path))
            ax[i][j].axis("off")
    fig.tight_layout()
    fig.savefig('evaluation/fig/{}.png'.format(args.input_root_gen.split('/')[-2]))






if __name__ == "__main__": 
    import os
    import pdb
    import numpy as np
    # if os.path.isdir('/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/sdxl/val2017_resize_256'):
    #     shutil.rmtree('/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/sdxl/val2017_resize_256')
    # os.makedirs('/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/sdxl/val2017_resize_256')
    def str2bool(str):
        return True if str.lower() == 'true' else False 

    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str ,default ='outputs/vae_decoder_only_l1-pruner_ratio-0.5',help='For pruner model only')  #/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools
    parser.add_argument("--input_dir", type=str ,default ='evaluation/coco2017/sdxl',help='directory stored generated image directory')
    parser.add_argument("--input_root_real", type=str ,default ='/mnt/bn/bytenn-yg2/datasets/coco2017_resize',help='real image directory')
    parser.add_argument("--input_root_gen", type=str, default ='evaluation/coco2017/sdxl/val2017_resize_gen',help='generated image directory')
    # parser.add_argument("--input_root_encoder", type=str, default ='/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize_encoder')
    # parser.add_argument("--input_root_rec", type=str, default ='/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize_decoder_rec')
    parser.add_argument("--if_baseline", type=str2bool, default = True ,help='if model is after-pruned model')
    parser.add_argument("--model_name", type=str, default =None, help='vae model name')
    parser.add_argument("--ckpt_name", type=str, default =None, help='model ckpt name')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device_ids", type=str, default="4, 5, 6, 7")
    parser.add_argument("--if_fp16", type=bool, default=True)
    args = parser.parse_args()

    # args.input_root_real = '/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize_5' 


    ################ sd1.5,fp16
    # args.input_dir = 'evaluation/coco2017/sd1.5_fp16'
    # args.input_root_gen = 'evaluation/coco2017/sd1.5_fp16/val2017_gen'



    ################ tiny taesd
    # args.input_dir = 'evaluation/coco2017/taesd_fp32'
    # args.input_root_gen = 'evaluation/coco2017/taesd_fp32/val2017_gen'


    ################ 剪枝模型
    # args.if_baseline=False
    # args.pretrained_model_name_or_path='outputs/vae_decoder_only_l1-pruner_ratio-0.5/checkpoint-50000'
    # args.input_dir = 'evaluation/coco2017/vae_decoder_only_l1-pruner_ratio-0.5_50000_fp16'
    # args.input_root_gen = 'evaluation/coco2017/vae_decoder_only_l1-pruner_ratio-0.5_50000_fp16/val2017_gen'

    # ############### 保存预测图
    torch.cuda.empty_cache()
    gc.collect()
    print('开始inference')
    inference_gen_dir_gpu(args)


    

    # ############### 计算指标
    # torch.cuda.empty_cache()
    # gc.collect()
    # print(10*'#' + 'index' + 10*'#')
    # main(args.input_root_real,args.input_root_gen,args)



    ############### 画图对比
    # args.input_root_gen = 'evaluation/coco2017/vae_decoder_only_l1-pruner_ratio-0.5_50000_fp16/val2017_gen'
    # args.input_root_gen = 'evaluation/coco2017/sd1.5/val2017_gen'
    #args.input_root_gen = 'evaluation/coco2017/taesd_fp32/val2017_gen'
    #plot_grid(args)
