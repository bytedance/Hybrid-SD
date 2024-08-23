import bson
from dataloader import KVReader
import io
from PIL import Image
import os
from torch.utils.data import Dataset


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

read_hdfs()




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