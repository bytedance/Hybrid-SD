
import os
import argparse
import zipfile
import subprocess
import time
from compression.utils.misc import change_img_size

# ZIP_URL="http://images.cocodataset.org/zips/val2014.zip"
ZIP_URL="http://images.cocodataset.org/zips/val2017.zip"


def downloads_stats(output_path):
    if not os.path.exists(os.path.join(output_path, 'val2017.zip')):
        t0 = time.time()
        print(f"Download the zip file from {ZIP_URL}")
        command = f"wget {ZIP_URL} -P {output_path}"
        subprocess.call(command, shell=True)
        print(f"** {time.time()-t0} sec elapsed") 

    img_dir = os.path.join(output_path, 'val2017')
    if not os.path.exists(img_dir):
        t0 = time.time()
        print(f"Unzip to {img_dir}")
        with zipfile.ZipFile(os.path.join(output_path,'val2017.zip'), 'r') as f:
            f.extractall(output_path)
        print(f"** {time.time()-t0} sec elapsed") 



# def resize(output_path):
#     img_dir =  os.path.join(output_path, 'val2017_real_10')
#     img_list = sorted([file for file in os.listdir(img_dir) if file.endswith('.jpg')])
#     if len(img_list) != 10:
#         raise ValueError(f"the number of images {len(img_list)} is something wrong; 5000 is expected")    
#     output_dir = os.path.join(output_path, 'val2017_real_10_resize')
#     if not os.path.exists(output_dir):       
#         os.makedirs(output_dir, exist_ok=True)
    
#     print(f"Resize to 512x512: {output_dir}")
#     change_img_size(img_dir, output_dir, resz=512)

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default="evaluation/coco2017")   
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    downloads_stats(args.save_dir)

          
if __name__ == "__main__":
    main()


