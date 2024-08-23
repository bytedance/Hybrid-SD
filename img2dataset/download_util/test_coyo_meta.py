import pandas as pd
import os
import os.path as osp
import tensorflow as tf

# Read the .data and .index files
path = "/mnt/bn/bytenn-data2/datasets/coyo_700m/coyo_700m_512plus_tags_filters_buckets"

# data = pd.read_csv(osp.join(path, "07_5005_448-576_part_0051_worker_2.data-00000-of-00002"), header=None)
# index = pd.read_csv(osp.join(path, "07_5005_448-576_part_0051_worker_2.index"), header=None)



# Read the .data-00000-of-00001 file
dataset = tf.data.TFRecordDataset(osp.join(path, "07_5005_448-576_part_0051_worker_2.data-00000-of-00002"))

# Iterate over the records in the dataset
import pdb;pdb.set_trace()
for record in dataset:
    # Parse the record
    example = tf.train.Example()
    example.ParseFromString(record.numpy())
    
    # Access the data in the record
    feature = example.features.feature['my_feature']
    value = feature.float_list.value[0]
    print(value)