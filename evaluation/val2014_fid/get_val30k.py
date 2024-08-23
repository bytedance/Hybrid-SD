import pandas as pd
import shutil
import os

# Paths
csv_file = '/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_30k/metadata.csv'
source_path = '/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_41k_full/val2014'
destination_path = '/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_41k_full/val2014_30K'

# Create the destination path if it doesn't exist
os.makedirs(destination_path, exist_ok=True)

# Read the CSV file
df = pd.read_csv(csv_file, header=None)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    file_name = row[0]  # Assuming the filename is in the first column
    
    if file_name.endswith('.jpg'):
        source_file = os.path.join(source_path, file_name)
        destination_file = os.path.join(destination_path, file_name)
        
        try:
            shutil.copy(source_file, destination_file)
            print(f"Copied {file_name} to {destination_path}")
        except FileNotFoundError:
            print(f"File {file_name} not found in {source_path}")