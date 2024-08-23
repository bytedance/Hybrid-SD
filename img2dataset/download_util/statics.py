from pandas import read_json
import os
#path = "/mnt/bn/bytenn-data2/liuhj/Laion_aesthetics_5plus_1024_33M"
#path = "/mnt/bn/bytenn-data2/liuhj/coyo-700m/coyo_data"
#path = "/mnt/bn/bytenn-data2/liuhj/test_spark"
path = "/mnt/bn/bytenn-yg2/datasets/Laion_aesthetics_5plus_1024_33M/Laion33m_data_test"

count = 0
i=0
for item in os.listdir(path):
    i+=1
    #print("index", i)
    if "_stats.json" in item:
        tar_json = read_json(os.path.join(path, item))
        count += tar_json["successes"][0]
        print(tar_json["successes"][0])
print(count)

# Laion 33M 13,368,611

# Laion 33M 11,757,077

# coyo      56,215,594

# Laion 33M test 13,435,094