import os
import glob
import cv2
import torchvision.transforms as transforms
import concurrent.futures
import unprocess
from collections import defaultdict

def process_image(image_path, shot_noise, read_noise, rgb2cam ,rgb_gain, red_gain, blue_gain ):
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    transform = transforms.ToTensor()
    image_tensor = transform(image)

    inv_image, metadata = unprocess.unprocess_event(image_tensor, rgb2cam ,rgb_gain, red_gain, blue_gain, low_light_factor=1)
    noisy_img = inv_image  # 取消噪声注入操作，效果差

    outimage_savepath = image_path.replace(input_path, output_path)
    outimage_savepath_dir = os.path.dirname(outimage_savepath)
    os.makedirs(outimage_savepath_dir, exist_ok=True)

    noisy_img = noisy_img.permute(1, 2, 0).numpy()
    noisy_img = (noisy_img * 255).clip(0, 255).astype('uint8')
    noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(outimage_savepath, noisy_img)


input_path = '/mnt/dataset/GOPRO/RGB_Dense/'
output_path = '/mnt/dataset/GOPRO/Dense_inv_isp/'

def process_images_in_folder(image_paths):
    shot_noise, read_noise = unprocess.random_noise_levels(min_shot=0.001, max_shot=0.012, min_read=0.0, max_read=0.26)
    rgb2cam = unprocess.random_ccm()
    rgb_gain, red_gain, blue_gain = unprocess.random_gains()
    for image_path in image_paths:
        process_image(image_path, shot_noise, read_noise, rgb2cam ,rgb_gain, red_gain, blue_gain  )

# 根据文件夹分组图像路径
folder_to_image_paths = defaultdict(list)
for image_path in glob.glob(input_path + "/*/*/*.png"):
    folder_path = os.path.dirname(image_path)
    folder_to_image_paths[folder_path].append(image_path)

os.makedirs(output_path, exist_ok=True)

# 为每个文件夹创建一个线程
with concurrent.futures.ThreadPoolExecutor() as executor:
    for folder, paths in folder_to_image_paths.items():
        executor.submit(process_images_in_folder,  paths)
print("处理完成，保存到", output_path)

