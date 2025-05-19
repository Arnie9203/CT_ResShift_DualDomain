#!/usr/bin/env python3
# check_shapes.py
# 这个脚本会读取所有指定的 .npy 文件，打印它们的维度到 shapes.txt

import numpy as np

# 在这里把你所有的文件路径粘贴到这个列表里：
file_paths = [
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_32views/test/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_32views/train/L067_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_32views/val/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_64views/test/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_64views/train/L067_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_64views/val/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_128views/test/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_128views/train/L067_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_128views/val/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_gt/test/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_gt/train/L067_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_img_gt/val/L310_FD_3_1_0004.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_32views/test/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_32views/train/L067_FD_3_1_0002.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_32views/val/L310_FD_3_1_0011.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_64views/test/L310_FD_3_1_0003.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_64views/train/L067_FD_3_1_0017.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_64views/val/L310_FD_3_1_0010.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_128views/test/L310_FD_3_1_0009.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_128views/train/L067_FD_3_1_0009.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_128views/val/L310_FD_3_1_0010.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_gt_fullviews/test/L310_FD_3_1_0001.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_gt_fullviews/train/L067_FD_3_1_0003.npy",
    "/home/arnie/Desktop/CT_Datasets/AAPM/3mm B30/all_sino_gt_fullviews/val/L310_FD_3_1_0010.npy",
    # …如果还有其他路径，继续添加
]

output_file = "shapes.txt"

with open(output_file, "w") as out_f:
    for path in file_paths:
        try:
            arr = np.load(path)
            shape = arr.shape
            out_f.write(f"{path}\t{shape}\n")
        except Exception as e:
            out_f.write(f"{path}\tError: {e}\n")

print(f"已将所有文件的维度写入 {output_file}")
