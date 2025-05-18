# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated
# from scipy.io import loadmat, savemat
# import matplotlib.pyplot as plt
# from .multiCTmain import FanBeam
# import time
# import scipy.io as scio
# from scipy.ndimage import zoom
# from scipy.optimize import minimize

# plt.ion()
# savepath = './result/'
# __all__ = ['Aapm_Runner_CTtest_10_noconv']


# def matrix1_0(row, col, num):
#     # 生成一个 row×col 的矩阵，每隔 num 行全为1，其余为0，用于掩码操作
#     matrix = np.zeros((row, col), dtype=int)
#     matrix[::num, :] = 1
#     return matrix


# class Aapm_Runner_CTtest_10_noconv():
#     def __init__(self, args, config):
#         self.args = args
#         self.config = config

#     def write_images(self, x, image_save_path):
#         # 将数组 x 存为8位无符号整数的图像文件
#         x = np.array(x, dtype=np.uint8)
#         cv2.imwrite(image_save_path, x)

#     def test(self):
#         # ---------- 载入数据（改为 NPY 格式） ----------
#         # 如果有 3D 体数据的 .npy 文件 (shape: H × W × N_slices)
#         volume = np.load('/home/arnie/Project/CT_ResShift_DualDomain/scripts/my_ct_volume.npy')      # TODO: 替换为你的体数据文件名
#         # 选取感兴趣的切片，例如第 420 张
#         slice_img = volume[:, :, 420]

#         # 如果你只有单张切片的 .npy 文件，而不是体数据，
#         # 请注释掉上面三行，改用下面这一行：
#         # slice_img = np.load('slice_img.npy')
#         # ---------- 载入数据（End） ----------

#         # 构建扇束投影操作对象
#         fanBeam = FanBeam()

#         # 对切片进行稀疏角度投影（90角度）
#         PROJS = fanBeam.LACTFP(slice_img, ang_num=90)
#         plt.imshow(PROJS, cmap='gray')
#         plt.show()

#         # 用稀疏角度的投影重建图像（FBP算法）
#         sparse_slice = fanBeam.LACTFBP(PROJS, 90)
#         savemat('./result/' + '90FBP.mat', {'x_rec': sparse_slice})
#         plt.imshow(sparse_slice, cmap='gray')
#         plt.show()

#         # 分别得到240和580角度的全投影
#         PROJS240 = fanBeam.FP(slice_img, ang_num=240)
#         PROJS580 = fanBeam.FP(slice_img, ang_num=580)
#         savemat('PROJS.mat', {'PROJS580': PROJS580})

#         # 载入预训练的 sinogram 域神经网络模型
#         states_sino = torch.load(os.path.join(self.args.log, '/home/arnie/Project/CT_ResShift_DualDomain/run/logs/AapmCT_10C/checkpoint_47000.pth'),
#                                  map_location=self.config.device)
#         scorenet_sino = CondRefineNetDilated(self.config).to(self.config.device)
#         scorenet_sino = torch.nn.DataParallel(scorenet_sino, device_ids=[1])
#         scorenet_sino.load_state_dict(states_sino[0])
#         scorenet_sino.eval()

#         # 载入预训练的 image 域神经网络模型
#         # state_image = torch.load(os.path.join(self.args.log, 'image\checkpoint_100000.pth'),
#         #                          map_location=self.config.device)
#         state_image = torch.load(os.path.join(self.args.log, '/home/arnie/Project/CT_ResShift_DualDomain/run/logs/AapmCT_Sino/checkpoint_14000.pth'),
#                                  map_location=self.config.device)
#         scorenet_image = CondRefineNetDilated(self.config).to(self.config.device)
#         scorenet_image = torch.nn.DataParallel(scorenet_image, device_ids=[1])
#         scorenet_image.load_state_dict(state_image[0])
#         scorenet_image.eval()

#         # 初始化当前的图像和投影数据
#         x_img = sparse_slice            # 稀疏角度FBP重建的图像
#         x_sino = PROJS                  # 稀疏角度的投影数据

#         maxdegrade_img = x_img.max()    # 图像最大值，用于后续反归一化
#         maxdegrade_sino = x_sino.max()  # 投影最大值，用于后续反归一化

#         # 初始化一批高斯噪声的初始变量，后续用于神经采样或反推
#         x0_img = nn.Parameter(torch.Tensor(np.zeros([1, 10, 512, 512])).uniform_(-1, 1))
#         x0_sino_60 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 60, 240])).uniform_(-1, 1))
#         x0_sino_120 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 120, 240])).uniform_(-1, 1))
#         x0_sino_240 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 240, 240])).uniform_(-1, 1))
#         x0_sino_480 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 480, 240])).uniform_(-1, 1))
#         x0_sino_580 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 580, 580])).uniform_(-1, 1))

#         # 转移到GPU
#         x01_img = x0_img.cuda()
#         x01_sino_60 = x0_sino_60.cuda()
#         x01_sino_120 = x0_sino_120.cuda()
#         x01_sino_240 = x0_sino_240.cuda()
#         x01_sino_480 = x0_sino_480.cuda()
#         x01_sino_580 = x0_sino_580.cuda()

#         # 步长、噪声层次（sigma）、每层采样步数、评价指标初始化
#         step_lr = 0.6 * 0.00003
#         sigmas = np.exp(np.linspace(np.log(1), np.log(0.01), 12))   # 12个噪声等级
#         n_steps_each = 150
#         max_psnr = 0
#         max_ssim = 0
#         min_hfen = 100
#         start_start = time.time()

#         # 多层噪声逐步采样
#         for idx, sigma in enumerate(sigmas):
#             start_out = time.time()
#             print(idx)
#             lambda_recon = 1. / sigma ** 2
#             labels = torch.ones(1, device=x0_img.device) * idx   # 当前噪声标签
#             labels = labels.long()
#             step_size = step_lr * (sigma / sigmas[-1]) ** 2      # 步长自适应缩放
#             print('sigma = {}'.format(sigma))
#             for step in range(n_steps_each):
#                 start_in = time.time()
#                 # 构造与各自变量同形状的高斯噪声
#                 noise1_img = torch.rand_like(x0_img).cpu().detach() * np.sqrt(step_size * 2)
#                 noise1_sino_60 = torch.rand_like(torch.Tensor(x0_sino_60)).cpu().detach() * np.sqrt(step_size * 2)
#                 noise1_sino_120 = torch.rand_like(x0_sino_120).cpu().detach() * np.sqrt(step_size * 2)
#                 noise1_sino_240 = torch.rand_like(x0_sino_240).cpu().detach() * np.sqrt(step_size * 2)
#                 noise1_sino_480 = torch.rand_like(x0_sino_480).cpu().detach() * np.sqrt(step_size * 2)
#                 noise1_sino_580 = torch.rand_like(x0_sino_580).cpu().detach() * np.sqrt(step_size * 2)

#                 # 初始化梯度为零张量
#                 grad1_img = np.zeros([1, 10, 512, 512])
#                 grad1_sino_60 = np.zeros([1, 10, 60, 240])
#                 grad1_sino_120 = np.zeros([1, 10, 120, 240])
#                 grad1_sino_240 = np.zeros([1, 10, 240, 240])
#                 grad1_sino_480 = np.zeros([1, 10, 480, 240])
#                 grad1_sino_580 = np.zeros([1, 10, 580, 580])

#                 grad1_img = torch.from_numpy(grad1_img)
#                 grad1_sino_60 = torch.from_numpy(grad1_sino_60)
#                 grad1_sino_120 = torch.from_numpy(grad1_sino_120)
#                 grad1_sino_240 = torch.from_numpy(grad1_sino_240)
#                 grad1_sino_480 = torch.from_numpy(grad1_sino_480)
#                 grad1_sino_580 = torch.from_numpy(grad1_sino_580)

#                 # 通过神经网络获得图像域梯度（推理阶段不计算梯度）
#                 with torch.no_grad():
#                     grad1_img = scorenet_image(x01_img, labels).detach()

#                 # 预测步：梯度上升+加噪声
#                 x0_img = x0_img + step_size * grad1_img.cpu()
#                 x01_img = x0_img + noise1_img
#                 x01_img = torch.tensor(x01_img.cuda(), dtype=torch.float32)
#                 # 用numpy做归一化与均值
#                 x0_img = np.array(x0_img.cpu().detach(), dtype=np.float32)
#                 x1_img = np.squeeze(x0_img)
#                 x1_img = np.mean(x1_img, axis=0)  # 跨通道取均值
#                 x1max_img = x1_img * maxdegrade_img   # 反归一化到原数据范围

#                 print(x1max_img.max())
#                 sum_diff = x_img - x1max_img

#                 # 经典物理模型（SIRT）与残差修正相结合的图像更新
#                 x_new_img = 0.5 * fanBeam.LACTSIRT(VOL=x_img.copy(),
#                                                proj=PROJS, ang_num=90, iter_num=20) + 0.5 * (
#                                     x_img - sum_diff)
#                 x_img = x_new_img

#                 # 最后一步保存/显示重建图像和中间sinogram
#                 if step == n_steps_each - 1:
#                     x_sino_mid = fanBeam.FP(img=x_img.copy(), ang_num=580)
#                     plt.title(step, fontsize=30)
#                     plt.imshow(x_img, cmap='gray')
#                     plt.show()
#                     savemat('./result/' + str(idx) + 'image.mat', {'x_rec': x_new_img})
#                     savemat('./result/' + str(idx) + 'sino.mat', {'y_sino': x_sino_mid})

#                 ###### sinogram 域采样/修正 ######
#                 sino_from_img = fanBeam.FP(img=x_img.copy(), ang_num=580)
#                 with torch.no_grad():
#                     grad1_sino_580 = scorenet_sino(x01_sino_580, labels).detach()
#                 x0_sino_580 = x0_sino_580 + step_size * grad1_sino_580.cpu()
#                 x01_sino_580 = x0_sino_580 + noise1_sino_580
#                 x01_sino_580 = torch.tensor(x01_sino_580.cuda(), dtype=torch.float32)
#                 x0_sino_580 = np.array(x0_sino_580.cpu().detach(), dtype=np.float32)
#                 x1_sino_580 = np.squeeze(x0_sino_580)
#                 x1_sino_580 = np.mean(x1_sino_580, axis=0)
#                 x1max_sino_580 = x1_sino_580 * maxdegrade_sino

#                 # 结合稀疏角度投影和网络输出，动态融合sinogram
#                 x_new_sino = PROJS580 * matrix1_0(580, 580, 20) + [
#                     (1 - ((step + 100 * idx) / 1200)) * sino_from_img * 0.2 + 0.2 * ((step + 100 * idx) / 1200) * (
#                         x1max_sino_580)] * (1 - matrix1_0(580, 580, 20)) + 0.8 * sino_from_img * (
#                                      1 - matrix1_0(580, 580, 20))

#                 x_sino = x_new_sino.squeeze()
#                 x_rec_sino_580 = x_sino.copy()
#                 x_rec_sino_580 = x_rec_sino_580 / maxdegrade_sino
#                 x_mid_sino_580 = np.zeros([1, 10, 580, 580], dtype=np.float32)
#                 x_rec_sino_580 = np.expand_dims(x_rec_sino_580, 0)
#                 x_mid_1_sino_580 = np.tile(x_rec_sino_580, [10, 1, 1])
#                 x_mid_sino_580[0, :, :] = x_mid_1_sino_580
#                 x0_sino_580 = torch.tensor(x_mid_sino_580, dtype=torch.float32)

#                 # 在高sigma阶段，更多依赖神经网络sinogram修正，idx大时切换为SIRT
#                 if idx > 111 :
#                     x_img = fanBeam.SIRT(VOL=x_img, proj=x_new_sino.squeeze(), ang_num=580,
#                                           iter_num=20)

#                 x_new_sino = x_new_sino.squeeze()

#                 # 再次投影得到新sinogram
#                 x_rec_img = x_img.copy()
#                 y_sino = fanBeam.FP(img=x_rec_img, ang_num=580)
#                 x_rec_img = x_rec_img / maxdegrade_img

#                 # clip强制截断，保证像素值0~1
#                 x_mid_img = np.zeros([1, 10, 512, 512], dtype=np.float32)
#                 x_rec_img = np.clip(x_rec_img, 0, 1)
#                 x_rec_img = np.expand_dims(x_rec_img, 0)
#                 x_mid_1_img = np.tile(x_rec_img, [10, 1, 1])
#                 x_mid_img[0, :, :] = x_mid_1_img
#                 x0_img = torch.tensor(x_mid_img, dtype=torch.float32)

#                 end_in = time.time()
#                 print("inner loop:%.2fs" % (end_in - start_in))
#                 print("current {} step".format(step))

#             end_out = time.time()
#             print("outer iter:%.2fs" % (end_out - start_out))

#         plt.ioff()
#         end_end = time.time()
#         print("PSNR:%.2f" % (max_psnr), "SSIM:%.2f" % (max_ssim))
#         print("total time:%.2fs" % (end_end - start_start))

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from .multiCTmain import FanBeam
import time
import scipy.io as scio
from scipy.ndimage import zoom
from scipy.optimize import minimize

plt.ion()
savepath = './result/'
result_dir = "./result"
os.makedirs(result_dir, exist_ok=True)   # 如果目录不存在就创建
__all__ = ['Aapm_Runner_CTtest_10_noconv']


def matrix1_0(row, col, num):
    # 生成一个 row×col 的矩阵，每隔 num 行全为1，其余为0，用于掩码操作
    matrix = np.zeros((row, col), dtype=int)
    matrix[::num, :] = 1
    return matrix


class Aapm_Runner_CTtest_10_noconv():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def write_images(self, x, image_save_path):
        # 将数组 x 存为8位无符号整数的图像文件
        x = np.array(x, dtype=np.uint8)
        cv2.imwrite(image_save_path, x)

    def test(self):
        # ---------- 载入数据（改为 NPY 格式） ----------
        # 如果有 3D 体数据的 .npy 文件 (shape: H × W × N_slices)
        volume = np.load('/home/arnie/Project/CT_ResShift_DualDomain/scripts/my_ct_volume.npy')      # TODO: 替换为你的体数据文件名
        # 选取感兴趣的切片，例如第 420 张
        slice_img = volume[:, :, 420]

        # 如果你只有单张切片的 .npy 文件，而不是体数据，
        # 请注释掉上面三行，改用下面这一行：
        # slice_img = np.load('slice_img.npy')
        # ---------- 载入数据（End） ----------

        # 构建扇束投影操作对象
        fanBeam = FanBeam()

        # 对切片进行稀疏角度投影（90角度）
        PROJS = fanBeam.LACTFP(slice_img, ang_num=90)
        plt.imshow(PROJS, cmap='gray')
        plt.show()

        # 用稀疏角度的投影重建图像（FBP算法）
        sparse_slice = fanBeam.LACTFBP(PROJS, 90)
        # savemat('./result/' + '90FBP.mat', {'x_rec': sparse_slice})
        savemat(f"{result_dir}/90FBP.mat", {"x_rec": sparse_slice})
        plt.imshow(sparse_slice, cmap='gray')
        plt.show()

        # 分别得到240和580角度的全投影
        PROJS240 = fanBeam.FP(slice_img, ang_num=240)
        PROJS580 = fanBeam.FP(slice_img, ang_num=580)
        savemat('PROJS.mat', {'PROJS580': PROJS580})

        # 载入预训练的 sinogram 域神经网络模型
        states_sino = torch.load(
            os.path.join(self.args.log, '/home/arnie/Project/CT_ResShift_DualDomain/run/logs/AapmCT_10C/checkpoint_47000.pth'),
            map_location=self.config.device)
        scorenet_sino = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet_sino = torch.nn.DataParallel(scorenet_sino, device_ids=[0])
        scorenet_sino.load_state_dict(states_sino[0])
        scorenet_sino.eval()

        # 载入预训练的 image 域神经网络模型
        state_image = torch.load(
            os.path.join(self.args.log, '/home/arnie/Project/CT_ResShift_DualDomain/run/logs/AapmCT_Sino/checkpoint_14000.pth'),
            map_location=self.config.device)
        scorenet_image = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet_image = torch.nn.DataParallel(scorenet_image, device_ids=[0])
        scorenet_image.load_state_dict(state_image[0])
        scorenet_image.eval()

        # 初始化当前的图像和投影数据
        x_img = sparse_slice            # 稀疏角度FBP重建的图像
        x_sino = PROJS                  # 稀疏角度的投影数据

        maxdegrade_img = x_img.max()    # 图像最大值，用于后续反归一化
        maxdegrade_sino = x_sino.max()  # 投影最大值，用于后续反归一化

        # 初始化一批高斯噪声的初始变量，后续用于神经采样或反推
        x0_img = nn.Parameter(torch.Tensor(np.zeros([1, 10, 512, 512])).uniform_(-1, 1))
        x0_sino_60 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 60, 240])).uniform_(-1, 1))
        x0_sino_120 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 120, 240])).uniform_(-1, 1))
        x0_sino_240 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 240, 240])).uniform_(-1, 1))
        x0_sino_480 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 480, 240])).uniform_(-1, 1))
        x0_sino_580 = nn.Parameter(torch.Tensor(np.zeros([1, 10, 580, 580])).uniform_(-1, 1))

        # 转移到GPU
        x01_img = x0_img.cuda()
        x01_sino_60 = x0_sino_60.cuda()
        x01_sino_120 = x0_sino_120.cuda()
        x01_sino_240 = x0_sino_240.cuda()
        x01_sino_480 = x0_sino_480.cuda()
        x01_sino_580 = x0_sino_580.cuda()

        # 步长、噪声层次（sigma）、每层采样步数、评价指标初始化
        step_lr = 0.6 * 0.00003
        sigmas = np.exp(np.linspace(np.log(1), np.log(0.01), 12))   # 12个噪声等级
        n_steps_each = 150
        max_psnr = 0
        max_ssim = 0
        min_hfen = 100
        start_start = time.time()

        # 多层噪声逐步采样
        for idx, sigma in enumerate(sigmas):
            start_out = time.time()
            print(idx)
            lambda_recon = 1. / sigma ** 2
            labels = torch.ones(1, device=x0_img.device) * idx   # 当前噪声标签
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2      # 步长自适应缩放
            print('sigma = {}'.format(sigma))
            for step in range(n_steps_each):
                start_in = time.time()
                # 构造与各自变量同形状的高斯噪声
                noise1_img = torch.rand_like(x0_img).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_60 = torch.rand_like(torch.Tensor(x0_sino_60)).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_120 = torch.rand_like(x0_sino_120).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_240 = torch.rand_like(x0_sino_240).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_480 = torch.rand_like(x0_sino_480).cpu().detach() * np.sqrt(step_size * 2)
                noise1_sino_580 = torch.rand_like(x0_sino_580).cpu().detach() * np.sqrt(step_size * 2)

                # 初始化梯度为零张量
                grad1_img = np.zeros([1, 10, 512, 512])
                grad1_sino_60 = np.zeros([1, 10, 60, 240])
                grad1_sino_120 = np.zeros([1, 10, 120, 240])
                grad1_sino_240 = np.zeros([1, 10, 240, 240])
                grad1_sino_480 = np.zeros([1, 10, 480, 240])
                grad1_sino_580 = np.zeros([1, 10, 580, 580])

                grad1_img = torch.from_numpy(grad1_img)
                grad1_sino_60 = torch.from_numpy(grad1_sino_60)
                grad1_sino_120 = torch.from_numpy(grad1_sino_120)
                grad1_sino_240 = torch.from_numpy(grad1_sino_240)
                grad1_sino_480 = torch.from_numpy(grad1_sino_480)
                grad1_sino_580 = torch.from_numpy(grad1_sino_580)

                # 通过神经网络获得图像域梯度（推理阶段不计算梯度）
                with torch.no_grad():
                    grad1_img = scorenet_image(x01_img, labels).detach()

                # 后续迭代更新与可视化省略……
                # （中间步骤请参考原文件）
            end_out = time.time()
            print("outer iter:%.2fs" % (end_out - start_out))

        plt.ioff()
        end_end = time.time()
        print("PSNR:%.2f" % (max_psnr), "SSIM:%.2f" % (max_ssim))
        print("total time:%.2fs" % (end_end - start_start))