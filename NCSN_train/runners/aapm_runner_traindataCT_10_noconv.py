

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
import torch.optim as optim
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from ..models.cond_refinenet_dilated_noconv import CondRefineNetDilated
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib.pyplot as plt
import pydicom as dicom
from .multiCTmain import FanBeam
import wandb
from itertools import zip_longest

# Multiprocessing for GPU domain split
import torch.multiprocessing as mp

# Get total GPU count and split for domain assignment
_TOTAL_GPUS = torch.cuda.device_count()
_HALF_GPUS = _TOTAL_GPUS // 2

# --- ❶ Dataset base directories (new) ---
IMG_BASE_DIR  = "/home/training_center/cyh_ct/CT/AAPM/3mm B30/all_img_gt"
SINO_BASE_DIR = "/home/training_center/cyh_ct/CT/AAPM/3mm B30/all_sino_gt_fullviews"

__all__ = ['AapmRunnerdata_10C']

# This FanBeam object seems unnecessary if data is already projected, but we'll keep it for context.
fanBeam = FanBeam()


# ============ NEW DATASET DEFINITIONS ============
class ImageDataset(Dataset):
    """Load image–domain data (.npy) and squeeze leading singleton dim."""
    def __init__(self, split: str):
        assert split in ("train", "val", "test")
        self.dir = os.path.join(IMG_BASE_DIR, split)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {self.dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)           # (1,H,W) or (H,W)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = np.squeeze(arr, axis=0)                           # -> (H,W)
        # 0‑1 normalisation
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)
        arr = np.expand_dims(arr, 2)                                # (H,W,1)
        arr = np.tile(arr, (1, 1, 10)).transpose(2, 0, 1)           # (10,H,W)
        return arr


class SinogramDataset(Dataset):
    """Load sinogram–domain full‑view data (.npy)."""
    def __init__(self, split: str):
        assert split in ("train", "val", "test")
        self.dir = os.path.join(SINO_BASE_DIR, split)
        self.files = sorted(glob.glob(os.path.join(self.dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {self.dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx]).astype(np.float32)           # (Nv,Nd)
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6)  # 0‑1
        arr = np.expand_dims(arr, 2)                                # (Nv,Nd,1)
        arr = np.tile(arr, (1, 1, 10)).transpose(2, 0, 1)           # (10,Nv,Nd)
        return arr
# ============ END DATASET DEFINITIONS ============

class AapmRunnerdata_10C():
    def __init__(self, args, config):
        self.args = args
        self.config = config

        # ============ Initialize wandb (Weights & Biases) for logging ============
        try:
            wandb.init(
                project=getattr(args, "wandb_project", "aapm-ct-diffusion"),
                entity=getattr(args, "wandb_entity", None),
                config={**vars(args), **vars(config)} if hasattr(args, '__dict__') else None,
                name=args.doc if hasattr(args, 'doc') else None,
                resume="allow",
                id=getattr(args, "wandb_run_id", None)
            )
            print("✅ [wandb] Initialized successfully.")
        except Exception as e:
            print(f"⚠️ [wandb] Initialization failed: {e}")

    def get_optimizer(self, parameters):
        if self.config.optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay,
                              betas=(self.config.optim.beta1, 0.999), amsgrad=self.config.optim.amsgrad)
        elif self.config.optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=self.config.optim.lr, weight_decay=self.config.optim.weight_decay)
        elif self.config.optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=self.config.optim.lr, momentum=0.9)
        else:
            raise NotImplementedError(f'Optimizer {self.config.optim.optimizer} not understood.')

    def logit_transform(self, image, lam=1e-6):
        image = lam + (1 - 2 * lam) * image
        return torch.log(image) - torch.log1p(-image)

    def _train_domain(self, rank: int):
        """
        Worker function for multiprocessing.spawn.
        rank 0: image-domain uses GPUs [0.._HALF_GPUS-1]
        rank 1: sinogram-domain uses GPUs [_HALF_GPUS.._TOTAL_GPUS-1]
        """
        # Assign subset of GPUs to this worker
        if rank == 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(_HALF_GPUS))
            dataset_cls = ImageDataset
            domain_name = "image"
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(_HALF_GPUS, _TOTAL_GPUS))
            dataset_cls = SinogramDataset
            domain_name = "sinogram"

        # Build dataloader
        loader = DataLoader(dataset_cls("train"),
                            batch_size=self.config.training.batch_size,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True)
        # Build model & optimizer, use all visible GPUs
        model = CondRefineNetDilated(self.config).to(self.config.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=None)
        optimizer = self.get_optimizer(model.parameters())

        # Noise schedule
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(self.config.model.sigma_begin),
                               np.log(self.config.model.sigma_end),
                               self.config.model.num_classes))
        ).float().to(self.config.device)

        step = 0
        for epoch in range(self.config.training.n_epochs):
            for batch in loader:
                step += 1
                X = batch.to(self.config.device) / 256.0 * 255.0 + torch.rand_like(batch) / 256.0
                if self.config.data.logit_transform:
                    X = self.logit_transform(X)
                labels = torch.randint(0, len(sigmas), (X.size(0),), device=X.device)

                # DSM loss for both domains
                loss = anneal_dsm_score_estimation(model, X, labels, sigmas, self.config.training.anneal_power)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % self.config.training.snapshot_freq == 0:
                    ckpt = [model.module.state_dict(), optimizer.state_dict(), step]
                    path = os.path.join(self.args.log, f'checkpoint_{domain_name}_{step}.pth')
                    torch.save(ckpt, path)
                if step >= self.config.training.n_iters:
                    return

    def train(self):
        """
        Launch two parallel processes: 
         - rank 0: image-domain on GPUs [0..half-1]
         - rank 1: sinogram-domain on GPUs [half..total-1]
        """
        mp.spawn(self._train_domain, nprocs=2)