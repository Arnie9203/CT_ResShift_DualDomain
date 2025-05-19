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
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import glob

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
        # ---------- 载入数据（读取测试文件夹中的所有 .npy 文件） ----------
        test_dir = '/home/arnie/Project/CT_ResShift_DualDomain/scripts/test'
        npy_files = sorted(glob.glob(os.path.join(test_dir, '*.npy')))
        for npy_file in npy_files:
            slice_img = np.load(npy_file)
            file_basename = os.path.basename(npy_file)

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

            # 初始化PSNR和SSIM列表，用于记录每个样本的指标
            PSNR_All = []
            SSIM_All = []
            # 定义日志保存目录和文件
            log_dir = os.path.join(result_dir, 'test_logs')
            os.makedirs(log_dir, exist_ok=True)
            test_log_file = os.path.join(log_dir, 'test_log.txt')
            # 写入表头
            with open(test_log_file, 'w') as f:
                f.write('File\t\tPSNR\t\tSSIM\n')

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

                    # 计算当前样本的PSNR和SSIM
                    output_np = x01_img.cpu().numpy()[0, 0, :, :]
                    label_np = x_img
                    p = psnr(output_np, label_np, data_range=1)
                    s = ssim(output_np, label_np, data_range=1)
                    PSNR_All.append(p)
                    SSIM_All.append(s)

                    # 将当前样本结果写入日志
                    with open(test_log_file, 'a') as f:
                        f.write(f'{file_basename}\t{p:.12f}\t{s:.12f}\n')

                    # 后续迭代更新与可视化省略……
                    # （中间步骤请参考原文件）
                end_out = time.time()
                print("outer iter:%.2fs" % (end_out - start_out))

        # 计算并打印平均和方差
        avg_psnr = np.mean(PSNR_All)
        avg_ssim = np.mean(SSIM_All)
        psnr_var = np.var(PSNR_All)
        ssim_var = np.var(SSIM_All)
        print(f"Average PSNR: {avg_psnr:.4f}, Variance: {psnr_var:.4f}")
        print(f"Average SSIM: {avg_ssim:.4f}, Variance: {ssim_var:.4f}")
        # 将结果追加写入日志
        with open(test_log_file, 'a') as f:
            f.write(f'Average PSNR: {avg_psnr:.4f}, Variance: {psnr_var:.4f}\n')
            f.write(f'Average SSIM: {avg_ssim:.4f}, Variance: {ssim_var:.4f}\n')

        plt.ioff()
        end_end = time.time()
        print("PSNR:%.2f" % (max_psnr), "SSIM:%.2f" % (max_ssim))
        print("total time:%.2fs" % (end_end - start_start))