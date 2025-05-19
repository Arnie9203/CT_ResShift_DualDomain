# aapm_runner_traindataCT_10_noconv.py
# 该脚本实现基于扩散模型的CT图像重建训练与测试流程
# 包括数据加载、模型初始化、训练循环、评估与结果保存
# -------------------- 第三方库导入 --------------------
import numpy as np
import tqdm
from ..losses.dsm import anneal_dsm_score_estimation
from ..losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import matplotlib.pyplot as plt
import random
import pydicom as dicom
import wandb  # <<< 新增

# -------------------- Python标准库导入 --------------------
import os
import glob
import glob
import scipy.io as scio
import shutil

# import tensorboardX
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# -------------------- 内部模块导入 --------------------
from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated, CondRefineNetDeeperDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
from .multiCTmain import FanBeam

__all__ = ['AapmRunnerdata_10C']
fanBeam = FanBeam()
randomData = [60, 120, 240]


# 类 DatasetLoader：同时加载输入图像、标签图像和投影数据文件路径列表，返回对应的三元组
class DatasetLoader(Dataset):
    # __init__：接受输入、标签、投影数据根目录，收集所有 .npy 文件路径并校验数量一致性
    def __init__(self, input_root, label_root, prj_root):
        self.input_files = sorted(glob.glob(os.path.join(input_root, '*.npy')))
        self.label_files = sorted(glob.glob(os.path.join(label_root, '*.npy')))
        self.prj_files   = sorted(glob.glob(os.path.join(prj_root,   '*.npy')))
        if not (len(self.input_files) == len(self.label_files) == len(self.prj_files)):
            raise ValueError("Input, label and proj file counts do not match")

    # __getitem__：根据索引加载对应的 .npy 文件，返回 torch.Tensor 格式的数据
    def __getitem__(self, index):
        input_data = torch.from_numpy(np.load(self.input_files[index])).float()
        label_data = torch.from_numpy(np.load(self.label_files[index])).float()
        prj_data   = torch.from_numpy(np.load(self.prj_files[index])).float()
        return input_data, label_data, prj_data

    # __len__：返回数据集样本总数
    def __len__(self):
        return len(self.input_files)


# 函数 get_data_loaders：根据给定的输入/标签/投影根目录，创建训练集和验证集 DataLoader，支持比例划分
def get_data_loaders(input_root, label_root, prj_root, batch_size, val_split=0.1):
    dataset = DatasetLoader(input_root, label_root, prj_root)
    size = len(dataset)
    indices = list(range(size))
    # 根据 val_split 比例划分训练集与验证集索引
    split = int(size * val_split)
    val_idx = indices[:split]
    train_idx = indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler   = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader   = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    return train_loader, val_loader


# 类 TestsetLoader：加载测试集数据，返回输入、标签、投影和文件名
class TestsetLoader(Dataset):
    def __init__(self, input_root, label_root, prj_root):
        self.input_files = sorted(glob.glob(os.path.join(input_root, '*.npy')))
        self.label_files = sorted(glob.glob(os.path.join(label_root, '*.npy')))
        self.prj_files   = sorted(glob.glob(os.path.join(prj_root,   '*.npy')))
        if not (len(self.input_files) == len(self.label_files) == len(self.prj_files)):
            raise ValueError("Input, label and proj file counts do not match")
    def __getitem__(self, index):
        input_data = torch.from_numpy(np.load(self.input_files[index])).float()
        label_data = torch.from_numpy(np.load(self.label_files[index])).float()
        prj_data   = torch.from_numpy(np.load(self.prj_files[index])).float()
        res_name   = os.path.basename(self.input_files[index])
        return input_data, label_data, prj_data, res_name
    def __len__(self):
        return len(self.input_files)


# 函数 get_test_loader：创建测试集 DataLoader，不打乱顺序
def get_test_loader(input_root, label_root, prj_root, batch_size):
    dataset = TestsetLoader(input_root, label_root, prj_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# 类 AapmRunnerdata_10C：封装模型训练与评估逻辑
class AapmRunnerdata_10C():
    # __init__：初始化参数、配置及 wandb 日志
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # 尝试初始化 Weights & Biases，用于实验追踪
        try:
            wandb.init(
                project=getattr(args, "wandb_project", "aapm-ct-diffusion"),
                entity=getattr(args, "wandb_entity", None),
                config={**vars(args), **vars(config)} if hasattr(args, '__dict__') else None,
                name=args.doc if hasattr(args, 'doc') else None,
                resume="allow",
                id=getattr(args, "wandb_run_id", None)
            )
            print("[wandb] Initialized successfully.")
        except Exception as e:
            print(f"[wandb] Initialization failed: {e}")

    # get_optimizer：根据配置选择并返回指定优化器
    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(self.config.optim.optimizer))

    # logit_transform：对输入图像进行 logit 变换，以匹配模型输入分布
    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    # train：训练主函数，负责数据预处理、加载、训练循环、定期评估与保存
    def train(self):
        # 构建训练/测试数据预处理管道
        if self.config.data.random_flip is False:
            tran_transform = test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])
        else:
            tran_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize(self.config.data.image_size),
                transforms.ToTensor()
            ])

        # 根据数据集类型选择数据加载方式
        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == "AAPM":
            print("train")
        # 获取 AAPM 数据集的训练、验证和测试 DataLoader
        input_root = os.path.join(self.args.run, 'datasets', 'input_npy')  # 调整为实际路径
        label_root = os.path.join(self.args.run, 'datasets', 'label_npy')
        prj_root   = os.path.join(self.args.run, 'datasets', 'prj_npy')
        train_loader, val_loader = get_data_loaders(input_root, label_root, prj_root, self.config.training.batch_size, val_split=0.1)
        test_loader = get_test_loader(input_root.replace('train', 'test'),
                                      label_root.replace('train', 'test'),
                                      prj_root.replace('train', 'test'),
                                      self.config.training.batch_size)

        test_iter = iter(test_loader)
        self.config.input_dim = self.config.data.image_size ** 2 * self.config.data.channels
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.doc)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        score = CondRefineNetDilated(self.config).to(self.config.device)
        score = torch.nn.DataParallel(score)
        optimizer = self.get_optimizer(score.parameters())

        # 恢复训练
        if self.args.resume_training:
            states = torch.load(os.path.join(self.args.log, 'checkpoint_10000.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            step = 43000
        else:
            step = 1

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        # 迭代训练多个 epoch
        for epoch in range(self.config.training.n_epochs):
            # 遍历一个 batch，执行前向、反向传播和优化
            for i, X in enumerate(train_loader):
                # print("🌶️")
                print(X[0].shape)
                step += 1
                score.train()
                X = tuple(torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x for x in X)
                X = tuple(x.to(self.config.device) for x in X)
                input_data = X[0]
                input_data = input_data / 256. * 255. + torch.rand_like(input_data) / 256.
                if self.config.data.logit_transform:
                    input_data = self.logit_transform(input_data)
                labels = torch.randint(0, len(sigmas), (input_data.shape[0],), device=input_data.device)
                if self.config.training.algo == 'dsm':
                    # print("👋👋👋 dsm")
                    loss = anneal_dsm_score_estimation(score, input_data, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    # print("👋👋👋 ssm")
                    loss = anneal_sliced_score_estimation_vr(score, input_data, labels, sigmas, n_particles=self.config.training.n_particles)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # wandb 记录 loss
                if wandb.run:
                    wandb.log({'epoch': epoch + 1, 'step': step, 'train_loss': loss.item()})
                logging.info("step: {}, loss: {}".format(step, loss.item()))

                if step >= self.config.training.n_iters:
                    if wandb.run:
                        wandb.finish()
                    return 0

                # 每隔固定步数在验证集上评估模型表现
                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X = next(test_iter)
                    test_X = tuple(torch.from_numpy(x).float() if isinstance(x, np.ndarray) else x for x in test_X)
                    test_X = tuple(x.to(self.config.device) for x in test_X)
                    test_input = test_X[0]
                    test_input = test_input / 256. * 255. + torch.rand_like(test_input) / 256.
                    if self.config.data.logit_transform:
                        test_input = self.logit_transform(test_input)
                    test_labels = torch.randint(0, len(sigmas), (test_input.shape[0],), device=test_input.device)
                    with torch.no_grad():
                        # print('💎💎💎')
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_input, test_labels, sigmas, self.config.training.anneal_power)
                        # print(f'🍺 score: {score}')
                    if wandb.run:
                        wandb.log({'step': step, 'test_loss': test_dsm_loss.item()})

                # 定期保存模型检查点
                if step % self.config.training.snapshot_freq == 0:
                    states = [
                        score.state_dict(),
                        optimizer.state_dict(),
                    ]
                    torch.save(states, os.path.join(self.args.log, 'checkpoint_{}.pth'.format(step)))
                    torch.save(states, os.path.join(self.args.log, 'checkpoint.pth'))
                    # wandb artifact（可选）
                    if wandb.run:
                        try:
                            artifact = wandb.Artifact(f'checkpoint_{step}', type='model')
                            artifact.add_file(os.path.join(self.args.log, f'checkpoint_{step}.pth'))
                            wandb.log_artifact(artifact)
                        except Exception as e:
                            print(f"[wandb] artifact failed: {e}")

        if wandb.run:
            wandb.finish()
        print("Training completed.")
