import os
import numpy as np
from tqdm import tqdm

# 输入和输出路径
input_dir = "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_32views/test"
output_dir = "/home/arnie/Desktop/npy_256x256_flattened_test"

# 创建目标文件夹
os.makedirs(output_dir, exist_ok=True)

# 遍历所有 .npy 文件
npy_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

for fname in tqdm(npy_files, desc="Processing"):
    path = os.path.join(input_dir, fname)
    data = np.load(path)

    if data.shape == (1, 256, 256):
        data = data.squeeze(0)  # 去掉最前面的 1 维 → (256, 256)
    elif data.shape != (256, 256):
        print(f"⚠️ 跳过 {fname}，因为维度是 {data.shape}")
        continue

    save_path = os.path.join(output_dir, fname)
    np.save(save_path, data)
