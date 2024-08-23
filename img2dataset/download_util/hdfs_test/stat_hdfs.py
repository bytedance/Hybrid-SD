import os
import bson
from dataloader import KVReader
import io
from PIL import Image

def read_hdfs(path):
    #path = '/mnt/bn/bytenn-data2/datasets/coyo_700m/coyo_64plus/8_10000_512-512_part_3119'
    #path = '/mnt/bn/bytenn-data2/laion2b/laion_2b_en_512plus_buckets/002555-10000-896-1152_00255_00002'
    num_parallel_reader = 12
    reader = KVReader(path, num_parallel_reader)
    keys = reader.list_keys() 
    #values = reader.read_many(keys)
    #data = bson.decode(values[0])
    return len(keys)

  



def find_index_files(directory):
    index_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.index'):
                index_files.append(os.path.join(root, file))
    return index_files

# Example usage
#current_dir = "/mnt/bn/bytenn-data2/datasets/coyo_700m/coyo_700m_512plus_tags_filters_buckets"
current_dir = "/mnt/bn/bytenn-data2/laion2b/laion_2b_en_512plus_buckets"
all_index_files = find_index_files(current_dir)
# print(len(all_index_files))

count = 0
for i, path in enumerate(all_index_files):
    count += read_hdfs(path.split('.')[0])
    print(i,count)
print("final count", count)
