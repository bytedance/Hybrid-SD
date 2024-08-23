import sys
sys.path.append("/mnt/bn/bytenn-data2/liuhj/pylib")
import collections
import os
import random
from tqdm import tqdm
import re
from torchvision.utils import save_image
import shutil
import  yaml
import json
import bson
from dataloader import KVReader
import io
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from transformers import CLIPTextModel, CLIPTokenizer
import bson
from dataloader import KVReader
import os



def read_hdfs():
    #path = '/mnt/bn/bytenn-data2/datasets/coyo_700m/coyo_64plus/8_10000_512-512_part_3119'
    path = '/mnt/bn/bytenn-data2/laion2b/laion_2b_en_512plus_buckets/002555-10000-896-1152_00255_00002'
    num_parallel_reader = 12
    reader = KVReader(path, num_parallel_reader)
    keys = reader.list_keys()  
    values = reader.read_many(keys[:10])
    j=0
    for i in values:
        j+=1
        data = bson.decode(i)
        img = data.pop('image')
        img = Image.open(io.BytesIO(img)).convert('RGB')
        img.save("./test_img/" + str(j) + ".jpg")
        print(img.size)
        print(data)
        if j>10:
            break






class Laion2B_en512plus_Dataset(Dataset):
    def __init__(self, data_dir="/mnt/bn/bytenn-data2/laion2b/laion_2b_en_512plus_buckets", transform=None):
        """
        "/mnt/bn/bytenn-data2/laion2b/laion_2b_en_512plus_buckets"
        """
        self.data_dir = data_dir
        self.transform = transform
        self.data_files =   []
        
        for index_path in os.listdir(data_dir):
            if index_path.endswith('.index'):
                self.data_files.append(os.path.join(data_dir, index_path))
                    

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data_files)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            A sample from the dataset.
        """
        # Load the data file
        file_path = os.path.join(self.data_dir, self.data_files[idx])
        data = self.load_data(file_path)

        # Apply any transformations
        if self.transform:
            data = self.transform(data)

        return data

    def load_data(self, file_path):
        """
        Load the data from a file.

        Args:
            file_path (str): Path to the data file.

        Returns:
            The loaded data.
        """
        # Implement your data loading logic here
        # This will depend on the format of your data files
        data = np.load(file_path)
        return data


    def get_data_indices(self):
        """
        Get the indices of the data items in the kv cache file.

        Returns:
            list: A list of (file_idx, item_idx) tuples.
        """
        data_indices = []
        with tarfile.open(self.data_tar_path, "r") as tar_file:
            for member in tar_file.getmembers():
                if member.name.endswith(".npy"):
                    file_idx = int(member.name.split("_")[1][:-4])
                    data_array = np.load(tar_file.extractfile(member))
                    for i in range(len(data_array)):
                        data_indices.append((file_idx, i))
        return data_indices
    
if __name__ == "__main__":
    read_hdfs()