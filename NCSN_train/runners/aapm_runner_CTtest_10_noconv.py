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
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import glob

plt.ion()
result_dir = "./result"
os.makedirs(result_dir, exist_ok=True)  # Create result directory if it doesn't exist
__all__ = ['Aapm_Runner_CTtest_10_noconv']

def matrix1_0(row, col, num):
    """Generates a mask matrix where every 'num'-th row is all ones."""
    matrix = np.zeros((row, col), dtype=int)
    if num > 0:
        matrix[::num, :] = 1
    return matrix

class Aapm_Runner_CTtest_10_noconv():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # =====================================================================
        # 1. Configuration Parameters
        # =====================================================================
        # Image and Sinogram dimensions (adapted from your data)
        IMG_SIZE = 256
        SPARSE_VIEWS = 32   # Sparse-angle view count
        FULL_VIEWS = 128    # Target full-angle view count
        DETECTORS = 512     # Detector count for sinogram

        # Langevin Dynamics parameters
        BATCH_CHANNELS = 10 # Number of parallel sampling chains
        STEP_LR = 0.6 * 0.00003
        N_STEPS_EACH = 150  # Steps per sigma level
        SIGMA_LEVELS = 12   # Number of noise levels

        # =====================================================================
        # 2. Setup Test Environment and Logging
        # =====================================================================
        test_dir = '/home/arnie/Project/CT_ResShift_DualDomain/scripts/test'
        npy_files = sorted(glob.glob(os.path.join(test_dir, '*.npy')))

        # Initialize lists to store metrics for all test files
        PSNR_all_files = []
        SSIM_all_files = []

        # Setup log file
        log_dir = os.path.join(result_dir, 'test_logs')
        os.makedirs(log_dir, exist_ok=True)
        test_log_file = os.path.join(log_dir, 'test_log.txt')
        with open(test_log_file, 'w') as f:
            f.write('File\t\t\tPSNR\t\t\tSSIM\n')

        # =====================================================================
        # 3. Load Models (once before the loop)
        # =====================================================================
        # Load sinogram domain model
        sino_model_path = '/home/arnie/Project/CT_ResShift_DualDomain/run/logs/AapmCT_Sino/checkpoint_14000.pth'
        states_sino = torch.load(sino_model_path, map_location=self.config.device)
        scorenet_sino = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet_sino = torch.nn.DataParallel(scorenet_sino, device_ids=[0])
        scorenet_sino.load_state_dict(states_sino[0])
        scorenet_sino.eval()

        # Load image domain model
        image_model_path = '/home/arnie/Project/CT_ResShift_DualDomain/run/logs/AapmCT_10C/checkpoint_47000.pth'
        state_image = torch.load(image_model_path, map_location=self.config.device)
        scorenet_image = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet_image = torch.nn.DataParallel(scorenet_image, device_ids=[0])
        scorenet_image.load_state_dict(state_image[0])
        scorenet_image.eval()
        
        # =====================================================================
        # 4. Main Loop: Process each test file
        # =====================================================================
        total_start_time = time.time()
        for npy_file in npy_files:
            file_basename = os.path.basename(npy_file)
            print(f"\n--- Processing file: {file_basename} ---")

            # Load ground truth image
            slice_img = np.load(npy_file)
            if slice_img.shape[0] != IMG_SIZE: # Ensure correct size
                zoom_factor = IMG_SIZE / slice_img.shape[0]
                slice_img = zoom(slice_img, zoom_factor)

            # --- Initial Data Generation ---
            fanBeam = FanBeam()
            PROJS_sparse = fanBeam.LACTFP(slice_img, ang_num=SPARSE_VIEWS)
            sparse_slice_fbp = fanBeam.LACTFBP(PROJS_sparse, SPARSE_VIEWS)
            PROJS_full_gt = fanBeam.FP(slice_img, ang_num=FULL_VIEWS) # Ground truth for data consistency

            # --- Initialize variables for reconstruction ---
            x_img = sparse_slice_fbp  # Start with FBP reconstruction
            maxdegrade_img = x_img.max() if x_img.max() > 0 else 1.0
            maxdegrade_sino = PROJS_sparse.max() if PROJS_sparse.max() > 0 else 1.0

            x0_img = nn.Parameter(torch.Tensor(np.zeros([1, BATCH_CHANNELS, IMG_SIZE, IMG_SIZE])).uniform_(-1, 1)).cuda()
            x0_sino = nn.Parameter(torch.Tensor(np.zeros([1, BATCH_CHANNELS, FULL_VIEWS, DETECTORS])).uniform_(-1, 1)).cuda()

            # --- Langevin Dynamics Loop ---
            sigmas = np.exp(np.linspace(np.log(1), np.log(0.01), SIGMA_LEVELS))
            
            for idx, sigma in enumerate(sigmas):
                print(f"  Sigma Level {idx + 1}/{len(sigmas)}: {sigma:.4f}")
                labels = torch.ones(1, device=x0_img.device).long() * idx
                step_size = STEP_LR * (sigma / sigmas[-1]) ** 2

                for step in range(N_STEPS_EACH):
                    # -- Image Domain Update --
                    with torch.no_grad():
                        grad_img = scorenet_image(x0_img, labels)
                    noise_img = torch.randn_like(x0_img) * np.sqrt(step_size * 2)
                    x0_img.data = x0_img.data + step_size * grad_img + noise_img
                    x1_img_est = np.mean(x0_img.cpu().detach().numpy().squeeze(), axis=0) * maxdegrade_img

                    # Image Data Consistency
                    sirt_recon = fanBeam.LACTSIRT(VOL=x_img.copy(), proj=PROJS_sparse, ang_num=SPARSE_VIEWS, iter_num=5)
                    x_img = 0.5 * sirt_recon + 0.5 * x1_img_est

                    # -- Sinogram Domain Update --
                    sino_from_img = fanBeam.FP(img=x_img.copy(), ang_num=FULL_VIEWS)
                    with torch.no_grad():
                        grad_sino = scorenet_sino(x0_sino, labels)
                    noise_sino = torch.randn_like(x0_sino) * np.sqrt(step_size * 2)
                    x0_sino.data = x0_sino.data + step_size * grad_sino + noise_sino
                    x1_sino_est = np.mean(x0_sino.cpu().detach().numpy().squeeze(), axis=0) * maxdegrade_sino

                    # Sinogram Data Consistency (using ground truth sparse views)
                    mask_interval = FULL_VIEWS // SPARSE_VIEWS
                    known_data_mask = matrix1_0(FULL_VIEWS, DETECTORS, mask_interval)
                    unknown_data_mask = 1 - known_data_mask
                    x_sino = PROJS_full_gt * known_data_mask + (0.5 * sino_from_img + 0.5 * x1_sino_est) * unknown_data_mask

                    # -- Feedback and Final Update for this step --
                    if idx > SIGMA_LEVELS // 2: # Use refined sinogram more in later stages
                        x_img = fanBeam.SIRT(VOL=x_img.copy(), proj=x_sino.squeeze(), ang_num=FULL_VIEWS, iter_num=5)

                    # -- Inject updated state back into samplers --
                    x_rec_img_norm = np.clip(x_img / maxdegrade_img, 0, 1)
                    x0_img.data = torch.tensor(np.tile(x_rec_img_norm[None, None, ...], (1, BATCH_CHANNELS, 1, 1)), dtype=torch.float32).cuda()
                    
                    x_rec_sino_norm = np.clip(x_sino / maxdegrade_sino, 0, 1)
                    x0_sino.data = torch.tensor(np.tile(x_rec_sino_norm[None, None, ...], (1, BATCH_CHANNELS, 1, 1)), dtype=torch.float32).cuda()

            # --- Save result and calculate metrics for the current file ---
            final_img_path = os.path.join(result_dir, f"{os.path.splitext(file_basename)[0]}_recon.png")
            # Normalize to 0-255 for saving as PNG
            recon_normalized = (x_img - x_img.min()) / (x_img.max() - x_img.min())
            cv2.imwrite(final_img_path, np.uint8(recon_normalized * 255))
            
            # Use original ground truth for metrics
            p = psnr(slice_img, x_img, data_range=slice_img.max() - slice_img.min())
            s = ssim(slice_img, x_img, data_range=slice_img.max() - slice_img.min())
            
            PSNR_all_files.append(p)
            SSIM_all_files.append(s)

            print(f"  File: {file_basename} -> PSNR: {p:.4f}, SSIM: {s:.4f}")
            with open(test_log_file, 'a') as f:
                f.write(f'{file_basename:<20}\t{p:.12f}\t{s:.12f}\n')

        # =====================================================================
        # 5. Final Report
        # =====================================================================
        avg_psnr = np.mean(PSNR_all_files)
        std_psnr = np.std(PSNR_all_files)
        avg_ssim = np.mean(SSIM_all_files)
        std_ssim = np.std(SSIM_all_files)
        
        summary_line1 = f"\nAverage PSNR: {avg_psnr:.4f} (+/- {std_psnr:.4f})"
        summary_line2 = f"Average SSIM: {avg_ssim:.4f} (+/- {std_ssim:.4f})"
        print(summary_line1)
        print(summary_line2)

        with open(test_log_file, 'a') as f:
            f.write('\n--- Summary ---\n')
            f.write(summary_line1 + '\n')
            f.write(summary_line2 + '\n')
            
        plt.ioff()
        total_end_time = time.time()
        print(f"\nTotal processing time: {(total_end_time - total_start_time)/60:.2f} minutes")