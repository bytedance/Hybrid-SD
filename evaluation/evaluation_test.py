
import pathlib
import torchvision.transforms as transforms
import pdb
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from cleanfid import fid
import numpy as np
from torchmetrics.aggregation import MeanMetric
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from scipy.stats import entropy
from PIL import Image
from tqdm import tqdm
import os
import pdb
import random
import dill
import shutil
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as SSIM
from pytorch_fid import fid_score
from datasets import load_dataset
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from compression.utils.misc import change_img_size
import cv2
from torchvision import utils as vutils
from torchvision.io import read_image
import lpips
import time
import argparse
# python3 -m pip install scikit-image==0.22.0
# python3 -m pip install pytorch_fid==0.3.0
# python3 -m pip install scipy==1.11.1
# python3 -m pip install lpips



def fid_eval(real_images_folder, generated_images_folder, batch_size=10,device='cuda:0'):
    fid_value = fid_score.calculate_fid_given_paths([real_images_folder, generated_images_folder],
    batch_size=batch_size,device=device,dims=2048,num_workers=1)
    return fid_value


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



def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)[..., :3]





def inception_score(dirPath, gen_data, batch_size, device, resize=True, splits=10):
    # Set up dtype
    device = torch.device(device)  # you can change the index of cuda
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).to(device)
    

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
 
    # Get predictions using pre-trained inception_v3 model
    print('Computing predictions using inception v3 model')
 
    files = readDir(dirPath)
    # N = gen_data.shape[0]
    N= len(files)
    preds = np.zeros((N, 1000))
    preds_gen = np.zeros((N, 1000))
    if batch_size > N:
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
 
    for i in tqdm(range(0, N, batch_size)):
        start = i
        end = i + batch_size
        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])
 
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255
 
        batch = torch.from_numpy(images).type(torch.FloatTensor)
        batch = batch.to(device)
        y = get_pred(batch)
        preds[i:i + batch_size] = get_pred(batch)


    # gen_data = gen_data.cpu().clone()
    # gen_data = gen_data.squeeze(0)
    # for i in tqdm(range(0, N, batch_size)):
    #     start = i
    #     end = i + batch_size
    #     batch = gen_data[start:end]
    #     batch = batch.to(device)
    #     y = get_pred(batch)
    #     preds_gen[i:i + batch_size] = get_pred(batch)

    # for 
    # pdb.set_trace()
    # laji = gen_data.mul_(255).clamp_(0, 255).numpy().astype(np.float32)/255
    # laji = torch.from_numpy(laji).type(torch.FloatTensor)
    # laji = laji.to(device)
    # aa = get_pred(laji)
    # pdb.set_trace()
    # toPIL = transforms.ToPILImage()
    # pic = toPIL(batch)
    # preds = gen_data # [10, 3, 512, 512]

    
    assert batch_size > 0
    assert N >= batch_size
    # Now compute the mean KL Divergence
    print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]  # split the whole data into several parts
        # part = part.numpy()
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(scores))
    
    return np.max(split_scores), np.mean(split_scores)



# def sp_score(file_real = './coco2017/val2017_real_10_resize', gen_path_dir = './coco2017/val2017_gen_10'):
#     path = readDir(file_real)
#     ssim = []
#     psnr = []
#     lpips_model = lpips.LPIPS(net="alex")
#     for i in path:
#         file_name = i.split('/')[-1]
#         real_image = imread(os.path.join(file_real,file_name))
#         gen_image = imread(os.path.join(gen_path_dir,file_name))
#         try:
#             ssim.append(SSIM(real_image, gen_image,win_size=11,multichannel=True,channel_axis=2))
#             psnr.append(compare_psnr(real_image, gen_image,data_range=255))
#         except:
#             pdb.set_trace()


#     dataset_real = load_dataset(file_real,split='train')
#     dataset_real = dataset_real.with_transform(transform) ## 变成tensor
#     dataset_real = dataset_real['image']
#     dataset_real = torch.stack(dataset_real)

#     dataset_gen = load_dataset(gen_path_dir,split='train')
#     dataset_gen = dataset_gen.with_transform(transform) ## 变成tensor
#     dataset_gen = dataset_gen['image']
#     dataset_gen = torch.stack(dataset_gen)
#     distance = lpips_model(dataset_real,dataset_gen)
    
#     return np.mean(ssim),np.mean(psnr),np.mean(np.squeeze(distance.detach().numpy()))
    



def transform(examples):
    val_transform = transforms.Compose([transforms.ToTensor()])
    examples['image'] = [val_transform(img.convert("RGB")) for img in examples['image']]
    return examples



# def generate_real_image_folder():
#     file_to_copy = readDir('./coco2017/val2017')
#     file_to_copy = random.sample(file_to_copy,10)
#     destination_directory = './coco2017/val2017_real_10'
#     if os.path.isdir(destination_directory):
#         shutil.rmtree(destination_directory)
#     os.makedirs(destination_directory)
#     for i in file_to_copy:
#         shutil.copy(i,destination_directory)



# def resize(input_dir='./coco2017/val2017', output_dir = './coco2017/val2017_gen'):
#     img_list = readDir(input_dir)
#     if len(img_list) != 5000:
#         raise ValueError(f"the number of images {len(img_list)} is something wrong; 5000 is expected")    
#     if os.path.exists(output_dir):       
#         shutil.rmtree(output_dir)
#     os.makedirs(output_dir, exist_ok=True)
#     preprocess = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.CenterCrop((512,512)),
#         transforms.Resize((512,512),interpolation=transforms.InterpolationMode.BILINEAR),
#         transforms.ToPILImage()
#     ])
#     for i in img_list:
#         save_path = os.path.join(output_dir,i.split('/')[-1].split('.')[0]+'.png')
#         img = Image.open(i)
#         img1 = preprocess(img.convert("RGB"))  ##torch.float32, torch.Size([3, 512, 512])
#         img1.save(save_path)



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
        batch_list = [img0, img1]
        return batch_list


def main(batch_size=32, file_real='./coco2017/val2017_real_10_resize',gen_path_dir='./coco2017/val2017_gen_10'):
    dataset = load_dataset(file_real,split='train')
    dataset = dataset.with_transform(transform)
    test = dataset['image']
    test = torch.stack(test)
    print('完成load dataset')

    ### gen_data = model(dataloader_save)
    ### 保存生成数据
    if os.path.exists(gen_path_dir):       
        shutil.rmtree(gen_path_dir)
    os.makedirs(gen_path_dir, exist_ok=True)
    file_real_list = readDir(file_real)
    gen_data = test.cpu().clone()
    preprocess = transforms.Compose([transforms.ToPILImage()])
    for i in range(len(file_real_list)):
        save_image = preprocess(gen_data[i])
        save_path = os.path.join(gen_path_dir,file_real_list[i].split('/')[-1].split('.')[0]+'.png')
        save_image.save(save_path,'PNG',lossless= True)


    ### 计算Inception score, 输入generated_images_folder
    MAX, IS = inception_score(gen_path_dir,gen_data, batch_size=batch_size, device='cuda:0',resize=True, splits=1)
    print(10*'-'+'The average IS is %.4f' % IS)

    # # ######## 计算fid,输入real_images_folder 和 generated_images_folder
    fid = fid_eval(file_real, gen_path_dir,batch_size=batch_size,device='cuda:0')
    print(10*'-'+'fid_value:',fid)

    # ##### 计算ssim和psnr, 输入real_image,和generate_image
    ssim, psnr, lps = sp_score(file_real, gen_path_dir)
    print(10*'-'+ 'ssim:',ssim)
    print(10*'-'+ 'psnr:',psnr)
    print(10*'-'+ 'lpips:',lps)




def main():
    psnr = PeakSignalNoiseRatio().to("cuda")
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to("cuda")
    ssim = StructuralSimilarityIndexMeasure().to("cuda")
    inception = InceptionScore().to("cuda")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--input_root_real", type=str ,default ='/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize_5')
    parser.add_argument("--input_root_gen", type=str, default ='/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize_5_gen')
    args = parser.parse_args()

    psnr_metric = MeanMetric()
    lpips_metric = MeanMetric()
    ssim_metric = MeanMetric()
    is_metric = MeanMetric()
    dataset = MultiImageDataset(args.input_root_real, args.input_root_gen)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    # make a json file
    progress_bar = tqdm(dataloader)
    with torch.inference_mode():
        for i, batch in enumerate(progress_bar):
            # to cuda
            batch = [img.to("cuda") for img in batch]
            batch_size = batch[0].shape[0]
            psnr_metric.update(psnr(batch[0].to(torch.float32), batch[1].to(torch.float32)).item(), batch_size)
            ssim_metric.update(ssim(batch[0].to(torch.float32), batch[1].to(torch.float32)).item(), batch_size)
            lpips_metric.update(lpips(batch[0] / 255, batch[1] / 255).item(), batch_size)
            is_metric.update(inception(batch[1]), batch_size)


    fid_score = fid.compute_fid(args.input_root0, args.input_root1)
    print("PSNR:", psnr_metric.compute().item())
    print("LPIPS:", lpips_metric.compute().item())
    print("ssim:", ssim_metric.compute().item())
    print("is:", is_metric.compute().item())
    print("FID:", fid_score)




# def inference_gen_dir(args):
#     # if torch.cuda.device_count() > 1:
#     #     print("Let's use", torch.cuda.device_count(), 'GPUs!')
#     #     model = torch.nn.DataParallel(model)
#     batch_size = len(args.device_ids)
#     if args.if_baseline:
#         from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
#         model = AutoencoderKL.from_pretrained('/mnt/bn/bytenn-yg2/pretrained_models/runwayml--stable-diffusion-v1-5', subfolder="vae").eval()
#         print(10*'#' + 'this is AutoencoderKL')

#         # from diffusers import AutoencoderTiny
#         # model = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).eval()
#         # print(10*'#' + 'this is AutoencoderTiny')
#     else:
#         from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
#         model = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae").eval()



#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     model.to(device)
#     tensor2img = transforms.Compose([transforms.ToPILImage()])
#     img2tensor = transforms.Compose([transforms.ToTensor()])
#     img_list = sorted([name for name in os.listdir(args.input_root_real) if name.endswith(".png")])
#     if os.path.isdir(args.input_root_gen):
#         shutil.rmtree(args.input_root_gen)
#     os.makedirs(args.input_root_gen)

#     a = time.time()
#     with torch.inference_mode():
#         for i in img_list:
#             read_path = os.path.join(args.input_root_real, i)
#             save_path = os.path.join(args.input_root_gen, i)
#             # imge_tensor = img2tensor(Image.open(read_path).convert("RGB"))
#             # imge_tensor = imge_tensor.unsqueeze(dim=0) 
#             imge_tensor = TF.to_tensor(Image.open(read_path).convert("RGB")).unsqueeze(0).to(device)
#             res = model(imge_tensor)['sample'].squeeze(0)
#             img = res.mul(255).clamp(0, 255).byte()
#             TF.to_pil_image(img).save(save_path)
#     print('inference成图片耗时:',time.time()-a)
#     flops, macs, params = calculate_flops(model=model, input_shape=(1, 3, 512, 512),output_as_string=True,output_precision=4,print_detailed=False,print_results=False)## decimal place
#     print('模型flops:',flops)
#     print('模型macs:',macs)
#     print('模型params:',params)




### 记得sort_file_path        
if __name__ == "__main__":
    main()
    ######### 制造测试数据 & resize
    # generate_real_image_folder() 
    # resize(input_dir='./coco2017/val2017', output_dir = './coco2017/val2017_resize')
    # a = time.time()
    # main(batch_size=128, file_real='./coco2017/val2017_resize',gen_path_dir='./coco2017/val2017_gen')
    # print('inference耗时s:',time.time()-a)
    # pdb.set_trace()
    ######### 制造生成数据 & evaluate
    
        



    ############ eval
    # from compression.optimize_vae.models.autoencoder_kl import AutoencoderKL
    # from compression.optimize_vae.webdata_hj import WebDataset
    # file_real = './coco2017/val2017_resize'
    # vae_tea = AutoencoderKL.from_pretrained('/mnt/bn/bytenn-data2/sd_models/runwayml--stable-diffusion-v1-5', subfolder="vae").eval()
    # test_dataset = load_dataset(file_real,split='train')
    # test_dataset = test_dataset.with_transform(transform)
    # test_dataset = test_dataset['image']
    # test_dataset = test_dataset.stack(test_dataset)
    # output = vae_tea(test_dataset)



# ----------The average IS is 5.7580
# ----------fid_value: -0.00019480877449495893
# ----------ssim: 1.0
# ----------psnr: inf
# ----------lpips: 0.0
#### check size
# for i in os.listdir('/mnt/bn/bytenn-data2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize'):
#     path = os.path.join('/mnt/bn/bytenn-data2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize',i)
#     if np.array(Image.open(path)).shape!=(512, 512, 3):
#         print(i)


########## 生成resize img
# img_list = sorted([os.path.join(args.input_root_real,name) for name in os.listdir(args.input_root_real) if name.endswith(".png")])
# test_dataset = load_dataset('imagefolder', data_files=img_list,split='train')
# test_dataset = test_dataset.with_transform(transform_256)['image'] ## 变成tensor 256
# # preprocess = transforms.Compose([transforms.ToPILImage()])
# for i in range(len(img_list)):
#     if i%100==0:
#         print('Now is:',i)
#     save_path = os.path.join('/mnt/bn/bytenn-yg2/pxr/bytenn_diffusion_tools/evaluation/coco2017/val2017_resize_256',img_list[i].split('/')[-1])
#     img = test_dataset[i]
#     TF.to_pil_image(img).convert('RGB').save(save_path)