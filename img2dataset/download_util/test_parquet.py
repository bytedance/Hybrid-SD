import sys
sys.path.append("/mnt/bn/bytenn-data2/liuhj/pylib")
from pandas import read_parquet

# pd = read_parquet("/mnt/bn/bytenn-data2/Laion_aesthetics_5plus_1024_33M/laion_aesthetics_1024_33M_1.parquet")
# pd = read_parquet("/mnt/bn/bytenn-data2/liuhj/coyo-700m/part-00000-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet")
pd = read_parquet("/mnt/bn/bytenn-data2/liuhj/MJ_dataset/meta_310k/train-00000-of-00001.parquet")
pd = read_parquet("/mnt/bn/bytenn-data2/liuhj/coyo-700m/coyo_meta_data/data/part-00059-17da4908-939c-46e5-91d0-15f256041956-c000.snappy.parquet")
pd.columns

for index, row in pd.iterrows():
    import pdb;pdb.set_trace()
    print(index)