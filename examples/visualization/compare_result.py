from PIL import Image, ImageOps, ImageDraw, ImageFont
import os

# 给定一系列包含图像的文件夹路径
root_dir = '/mnt/bn/bytenn-yg2/ycq/workspace/bytenn_diffusion_tools/publics/BK-SDM/results'
folder_paths = [f'{root_dir}/byte_sd1.5/im512', 
                f'{root_dir}/byte_sd1.5_deepcache_i3/im512', 
                f'{root_dir}/byte_sd1.5_deepcache_i5/im512', 
                f'{root_dir}/byte_sd1.5_tgate/im512',
                f'{root_dir}/byte_sd1.5_tome_r0.5/im512']
labels = ['Byte SD1.5', 'DeepCache(I=3)', 'DeepCache(I=5)', 'Tgate(s=8)', 'ToME(r=0.5)']

def add_white_border(img):
    border = (10, 10, 10, 10) # left, up, right, bottom
    return ImageOps.expand(img, border, fill='white')

def create_label(text, font_size, label_size, bg_color = "white"):
    # 创建一个新的图像作为标签
    W, H = label_size
    label = Image.new('RGB', label_size, bg_color)
    draw = ImageDraw.Draw(label)
    # w, h = draw.textsize(text)
    w, h = len(text) * font_size, font_size 
    draw.text(((W-w )/2,(H-h)/2), text, fill="black", font_size=font_size)
    # 在标签图像上添加文本
    # draw.text((0, 0), text, fill=label_color)
    
    return label

n_imgs = 8
images = [[add_white_border(Image.open(os.path.join(folder, img_file))) for img_file in sorted(os.listdir(folder))[:n_imgs]] for folder in folder_paths]
img_width, img_height = images[0][0].size

for i, label in enumerate(labels):
    label = create_label(label, font_size=35, label_size=(img_width, img_height))
    images[i].insert(0, label)

num_rows = len(images)
num_cols = len(images[0])
print(f'n_rows={num_rows}, n_cols={num_cols}')

# 创建一个新的空白图像，尺寸足以容纳所有拼接的图像
total_width = img_width * num_cols
total_height = img_height * num_rows
new_img = Image.new('RGB', (total_width, total_height))

# 拼接图像
for i in range(num_rows):
    for j in range(num_cols):
        new_img.paste(images[i][j], (j*img_height, i*img_width))

# 保存拼接后的图像
img_path = f'{root_dir}/combined_image.jpg'
new_img.save(img_path)
print(f"save img to {img_path}")


