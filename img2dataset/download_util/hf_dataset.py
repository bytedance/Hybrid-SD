import os
os.environ["http_proxy"] = "http://sys-proxy-rd-relay.byted.org:8118"
os.environ["https_proxy"] = "http://sys-proxy-rd-relay.byted.org:8118"
os.environ["no_proxy"] = ".byted.org"

from datasets import load_dataset
  
test_dataset = load_dataset("CortexLM/midjourney-v6", data_files="train-00000-of-00001.parquet")

test_dataset.save_to_disk("train-00000-of-00001.parquet")