import os
from PIL import Image

src_path="/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_41k_full/val2014_30K"

des_path="/mnt/bn/bytenn-yg2/datasets/mscoco_val2014_41k_full/val2014_30K_512"
os.makedirs(des_path, exist_ok=True)
def center_crop_and_resize(image, size):
    width, height = image.size
    new_width = new_height = min(width, height)
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((size, size), Image.ANTIALIAS)
    return image


for i in os.listdir(src_path):
    image_path = os.path.join(src_path, i)
    image = Image.open(image_path)
    image = center_crop_and_resize(image, 512)
    image.save(os.path.join(des_path, i))