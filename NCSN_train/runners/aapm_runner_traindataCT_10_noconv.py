import numpy as np
import tqdm
from ..losses.dsm import anneal_dsm_score_estimation
from ..losses.sliced_sm import anneal_sliced_score_estimation_vr
import torch.nn.functional as F
import logging
import torch
import os
import glob
import scipy.io as scio
import shutil
# import tensorboardX
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated, CondRefineNetDeeperDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
import random
import pydicom as dicom
from .multiCTmain import FanBeam
import wandb  # <<< 新增

__all__ = ['AapmRunnerdata_10C']
fanBeam = FanBeam()
randomData = [60, 120, 240]


class trainset_loader(Dataset):
    def __init__(self):
        data_dir = "/home/training_center/cyh_ct/CT/AAPM/3mm B30/all_img_gt/train"
        # /home/training_center/cyh_ct/CT/AAPM/3mm B30/all_sino_gt_fullviews
        self.files_A = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        print(f"[trainset_loader] Found {len(self.files_A)} samples in {data_dir}")
        if len(self.files_A) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

    def __getitem__(self, index):
        file_A = self.files_A[index]
        label_data = np.load(file_A).astype(np.float32)
        label_data = np.squeeze(label_data, axis=0)
        # proj = fanBeam.FP(label_data, 580)
        # label_data = proj
        data_array = (label_data - np.min(label_data)) / (np.max(label_data) - np.min(label_data))
        data_array = np.expand_dims(data_array, 2)
        data_array_10 = np.tile(data_array, (1, 1, 10))
        data_array_10 = data_array_10.transpose((2, 0, 1))
        return data_array_10

    def __len__(self):
        return len(self.files_A)


class testset_loader(Dataset):
    def __init__(self):
        data_dir = "/home/training_center/cyh_ct/CT/AAPM/3mm B30/all_img_gt/val"
        self.files_A = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        print(f"[testset_loader] Found {len(self.files_A)} samples in {data_dir}")
        if len(self.files_A) == 0:
            raise FileNotFoundError(f"No .npy files found in {data_dir}")

    def __getitem__(self, index):
        file_A = self.files_A[index]
        label_data = np.load(file_A).astype(np.float32)
        label_data = np.squeeze(label_data, axis=0)
        data_array = (label_data - np.min(label_data)) / (np.max(label_data) - np.min(label_data))
        data_array = np.expand_dims(data_array, 2)
        data_array_10 = np.tile(data_array, (1, 1, 10))
        data_array_10 = data_array_10.transpose((2, 0, 1))
        return data_array_10

    def __len__(self):
        return len(self.files_A)


class GetCT(Dataset):
    def __init__(self, root, augment=None):
        super().__init__()
        self.data_names = np.array([root + "/" + x for x in os.listdir(root)])
        print(f'self.data_names: {self.data_names}')
        self.augment = None

    def __getitem__(self, index):
        dataCT = dicom.read_file(self.data_names[index])
        data_array = dataCT.pixel_array.astype(np.float32) * dataCT.RescaleSlope + dataCT.RescaleIntercept
        data_array = (data_array - np.min(data_array)) / (np.max(data_array) - np.min(data_array))
        data_array = np.expand_dims(data_array, 2)
        data_array_10 = np.tile(data_array, (1, 1, 10))
        data_array_10 = data_array_10.transpose((2, 0, 1))
        return data_array_10

    def __len__(self):
        return len(self.data_names)


class AapmRunnerdata_10C():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # ============ wandb 初始化 ============
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

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def train(self):
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

        if self.config.data.dataset == 'CIFAR10':
            dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=True, download=True,
                              transform=tran_transform)
            test_dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10_test'), train=False, download=True,
                                   transform=test_transform)
        elif self.config.data.dataset == "AAPM":
            print("train")
        dataloader = DataLoader(trainset_loader(), batch_size=self.config.training.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(testset_loader(), batch_size=self.config.training.batch_size, shuffle=True, num_workers=0, drop_last=True)

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
            states = torch.load(os.path.join(self.args.log, 'checkpoint_43000.pth'))
            score.load_state_dict(states[0])
            optimizer.load_state_dict(states[1])
            step = 43000
        else:
            step = 1

        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin), np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))).float().to(self.config.device)

        for epoch in range(self.config.training.n_epochs):
            for i, X in enumerate(dataloader):
                print("🌶️")
                print(X.shape)
                step += 1
                score.train()
                X = torch.from_numpy(X).float() if isinstance(X, np.ndarray) else X
                X = X.to(self.config.device)
                X = X / 256. * 255. + torch.rand_like(X) / 256.
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)
                labels = torch.randint(0, len(sigmas), (X.shape[0],), device=X.device)
                if self.config.training.algo == 'dsm':
                    print("👋👋👋 dsm")
                    loss = anneal_dsm_score_estimation(score, X, labels, sigmas, self.config.training.anneal_power)
                elif self.config.training.algo == 'ssm':
                    print("👋👋👋 ssm")
                    loss = anneal_sliced_score_estimation_vr(score, X, labels, sigmas, n_particles=self.config.training.n_particles)

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

                # 每100 step进行评估
                if step % 100 == 0:
                    score.eval()
                    try:
                        test_X = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_loader)
                        test_X = next(test_iter)
                    test_X = torch.from_numpy(test_X).float() if isinstance(test_X, np.ndarray) else test_X
                    test_X = test_X.to(self.config.device)
                    test_X = test_X / 256. * 255. + torch.rand_like(test_X) / 256.
                    if self.config.data.logit_transform:
                        test_X = self.logit_transform(test_X)
                    test_labels = torch.randint(0, len(sigmas), (test_X.shape[0],), device=test_X.device)
                    with torch.no_grad():
                        print('💎💎💎')
                        test_dsm_loss = anneal_dsm_score_estimation(score, test_X, test_labels, sigmas, self.config.training.anneal_power)
                        print(f'🍺 score: {score}')
                    if wandb.run:
                        wandb.log({'step': step, 'test_loss': test_dsm_loss.item()})

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