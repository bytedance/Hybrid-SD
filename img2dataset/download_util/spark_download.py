from img2dataset import download
import shutil
import os
os.environ["http_proxy"] = "http://sys-proxy-rd-relay.byted.org:8118"
os.environ["https_proxy"] = "http://sys-proxy-rd-relay.byted.org:8118"
from pyspark.sql import SparkSession  # pylint: disable=import-outside-toplevel

output_dir = "/mnt/bn/bytenn-data2/liuhj/Laion_aesthetics_5plus_1024_33M_test"
output_dir = "/mnt/bn/bytenn-data2/liuhj/test_spark"
# local spark
# spark = (
#     SparkSession.builder.config("spark.driver.memory", "16G").master("local[16]").appName("spark-stats").getOrCreate()
# )

download(
    processes_count=16,
    thread_count=32,
    url_list="/mnt/bn/bytenn-data2/Laion_aesthetics_5plus_1024_33M",
    image_size=1024,
    output_folder=output_dir,
    output_format="webdataset",
    input_format="parquet",
    url_col="URL",
    caption_col="TEXT",
    enable_wandb=False,
    number_sample_per_shard=10000,
    distributor="pyspark",
    max_shard_retry=5,
)
