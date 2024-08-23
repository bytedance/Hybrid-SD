import sys
sys.path.append("/mnt/bn/bytenn-data2/liuhj/pylib")
from pyspark.sql import SparkSession

if __name__ == '__main__':
    # create spark ssession
    my_spark = SparkSession.builder.enableHiveSupport() \
    .appName("spark_common_process") \
    .config("spark.hadoop.mapreduce.output.fileoutputformat.compress", "false") \
    .config("spark.hadoop.mapreduce.map.output.compress", "false") \
    .getOrCreate()
    sc = my_spark.sparkContext


    df = my_spark.read.parquet('hdfs://haruna/home/byte_data_seed/hl_lq/iccv/liguodong/data/laion2B-en-joined_aesthetic-tags') # 读取文件，可以读取目录
    df = df.filter(df.aesthetic>0.8)

    # df.write.mode('overwrite').parquet('hdfs://') # 可以保存为parquet文件
    # df.write.mode('overwrite').json('hdfs://') # 可以保存为json文件，每行一条数据，格式为json string

    # 停止spark session
    my_spark.stop()