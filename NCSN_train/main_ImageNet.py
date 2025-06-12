import sys
import os
import numpy as np
from numpy import matlib
from .loadData import loadData
from scipy.interpolate import interpn, griddata, interp2d
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import cv2
import time
import astra
import scipy.io as sio
from .datamaking import datamaking_test
from torch.utils.data import DataLoader
from .compose import compose
import shutil

class FanBeam():
    def __init__(self):
        # --- projection geometries ---
        self.projGeom29     = astra.create_proj_geom('fanflat', 0.35, 580, np.linspace(0, np.pi,   29, endpoint=False), 500, 500)
        self.projGeom30     = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi,   30, endpoint=False), 500, 500)
        self.projGeom60     = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi,   60, endpoint=False), 500, 500)
        self.projGeom120    = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi,  120, endpoint=False), 500, 500)
        self.projGeom240    = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi,  240, endpoint=False), 500, 500)
        self.projGeom480    = astra.create_proj_geom('fanflat', 0.35, 240, np.linspace(0, np.pi,  480, endpoint=False), 500, 500)
        self.projGeom580    = astra.create_proj_geom('fanflat', 0.35, 580, np.linspace(0, np.pi,  580, endpoint=False), 500, 500)

        # few‐view 32/64/128 geometries with 512 channels
        self.projGeom32_512  = astra.create_proj_geom('fanflat', 0.35, 512, np.linspace(0, np.pi,  32, endpoint=False), 500, 500)
        self.projGeom64_512  = astra.create_proj_geom('fanflat', 0.35, 512, np.linspace(0, np.pi,  64, endpoint=False), 500, 500)
        self.projGeom128_512 = astra.create_proj_geom('fanflat', 0.35, 512, np.linspace(0, np.pi, 128, endpoint=False), 500, 500)

        # --- volume geometries ---
        # image-domain: 256×256 grid
        self.volGeom   = astra.create_vol_geom(256, 256, (-256/2), (256/2), (-256/2), (256/2))
        # sinogram-domain full-resolution: 512×512 grid
        self.volGeom512 = astra.create_vol_geom(512, 512, (-512/2), (512/2), (-512/2), (512/2))

        # LACT geometries
        self.projGeomLACT90 = astra.create_proj_geom('fanflat', 0.7, 1500,
            np.linspace(0, np.pi/2 + np.pi/12, 480, endpoint=False), 2000, 500)
        self.projGeomLACT60 = astra.create_proj_geom('fanflat', 1.5,  640,
            np.linspace(0, np.pi/3, 240, endpoint=False), 1500, 500)

    def FP(self, img, ang_num):
        """
        Forward project a 2D image 'img' at ang_num angles.
        Automatically picks 256×256 vs 512×512 volume grid based on img.shape.
        """
        # select projection geometry
        if   ang_num == 29:  projGeom = self.projGeom29
        elif ang_num == 30:  projGeom = self.projGeom30
        elif ang_num == 60:  projGeom = self.projGeom60
        elif ang_num == 120: projGeom = self.projGeom120
        elif ang_num == 240: projGeom = self.projGeom240
        elif ang_num == 480: projGeom = self.projGeom480
        elif ang_num == 580: projGeom = self.projGeom580
        elif ang_num == 32:  projGeom = self.projGeom32_512
        elif ang_num == 64:  projGeom = self.projGeom64_512
        elif ang_num == 128: projGeom = self.projGeom128_512
        else:
            raise ValueError(f"Unsupported angle count: {ang_num}")

        # choose volume geometry by input resolution
        if img.shape == (512, 512):
            volGeom = self.volGeom512
        else:
            volGeom = self.volGeom

        rec_id  = astra.data2d.create('-vol',  volGeom, img)
        proj_id = astra.data2d.create('-sino', projGeom)
        cfg     = astra.astra_dict('FP_CUDA')
        cfg['VolumeDataId']     = rec_id
        cfg['ProjectionDataId'] = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        pro = astra.data2d.get(proj_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return pro

    def FBP(self, proj, ang_num):
        """
        Filtered back-projection of sinogram 'proj' at ang_num angles.
        Automatically picks 256×256 vs 512×512 volume grid based on ang_num==580.
        """
        # select projection geometry
        if   ang_num == 29:  projGeom = self.projGeom29
        elif ang_num == 30:  projGeom = self.projGeom30
        elif ang_num == 60:  projGeom = self.projGeom60
        elif ang_num == 120: projGeom = self.projGeom120
        elif ang_num == 240: projGeom = self.projGeom240
        elif ang_num == 480: projGeom = self.projGeom480
        elif ang_num == 580: projGeom = self.projGeom580
        elif ang_num == 32:  projGeom = self.projGeom32_512
        elif ang_num == 64:  projGeom = self.projGeom64_512
        elif ang_num == 128: projGeom = self.projGeom128_512
        else:
            raise ValueError(f"Unsupported angle count: {ang_num}")

        # choose volume geometry
        if ang_num == 580:
            volGeom = self.volGeom512
        else:
            volGeom = self.volGeom

        rec_id  = astra.data2d.create('-vol',  volGeom)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg     = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']    = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def SIRT(self, VOL, proj, ang_num, iter_num):
        """
        SIRT iterative reconstruction.
        """
        # select projection geometry
        if   ang_num == 29:  projGeom = self.projGeom29
        elif ang_num == 30:  projGeom = self.projGeom30
        elif ang_num == 60:  projGeom = self.projGeom60
        elif ang_num == 120: projGeom = self.projGeom120
        elif ang_num == 240: projGeom = self.projGeom240
        elif ang_num == 480: projGeom = self.projGeom480
        elif ang_num == 580: projGeom = self.projGeom580
        elif ang_num == 32:  projGeom = self.projGeom32_512
        elif ang_num == 64:  projGeom = self.projGeom64_512
        elif ang_num == 128: projGeom = self.projGeom128_512
        else:
            raise ValueError(f"Unsupported angle count: {ang_num}")

        # choose volume geometry
        if ang_num == 580:
            volGeom = self.volGeom512
        else:
            volGeom = self.volGeom

        # create volume & projection data
        rec_id  = astra.data2d.create('-vol',  volGeom) if VOL is None else astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg     = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']    = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def EM(self, VOL, proj, ang_num, iter_num):
        """
        EM iterative reconstruction.
        """
        # select projection geometry (same as SIRT)
        if   ang_num == 29:  projGeom = self.projGeom29
        elif ang_num == 30:  projGeom = self.projGeom30
        elif ang_num == 60:  projGeom = self.projGeom60
        elif ang_num == 120: projGeom = self.projGeom120
        elif ang_num == 240: projGeom = self.projGeom240
        elif ang_num == 480: projGeom = self.projGeom480
        elif ang_num == 580: projGeom = self.projGeom580
        elif ang_num == 32:  projGeom = self.projGeom32_512
        elif ang_num == 64:  projGeom = self.projGeom64_512
        elif ang_num == 128: projGeom = self.projGeom128_512
        else:
            raise ValueError(f"Unsupported angle count: {ang_num}")

        # choose volume geometry
        if ang_num == 580:
            volGeom = self.volGeom512
        else:
            volGeom = self.volGeom

        rec_id  = astra.data2d.create('-vol',  volGeom) if VOL is None else astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg     = astra.astra_dict('EM_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']    = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def LACTFP(self, img, ang_num):
        """
        Forward projection for limited-angle CT (LACT).
        """
        # select LACT projection geometry
        if ang_num == 60:
            projGeom = self.projGeomLACT60
        elif ang_num == 90:
            projGeom = self.projGeomLACT90
        else:
            raise ValueError(f"Unsupported LACT angle: {ang_num}")

        # choose volume geometry by input resolution
        if img.shape == (512, 512):
            volGeom = self.volGeom512
        else:
            volGeom = self.volGeom

        rec_id  = astra.data2d.create('-vol',  volGeom, img)
        proj_id = astra.data2d.create('-sino', projGeom)
        cfg     = astra.astra_dict('FP_CUDA')
        cfg['VolumeDataId']     = rec_id
        cfg['ProjectionDataId'] = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        pro = astra.data2d.get(proj_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return pro

    def LACTSIRT(self, VOL, proj, ang_num, iter_num):
        """
        SIRT for limited-angle CT (LACT).
        """
        if   ang_num == 60:
            projGeom = self.projGeomLACT60
        elif ang_num == 90:
            projGeom = self.projGeomLACT90
        else:
            raise ValueError(f"Unsupported LACT angle: {ang_num}")

        # choose volume geometry
        if ang_num == 580:
            volGeom = self.volGeom512
        else:
            volGeom = self.volGeom

        rec_id  = astra.data2d.create('-vol',  volGeom) if VOL is None else astra.data2d.create('-vol', volGeom, VOL)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg     = astra.astra_dict('SIRT_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']    = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, iter_num)
        rec = astra.data2d.get(rec_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec

    def LACTFBP(self, proj, ang_num):
        """
        FBP for limited-angle CT (LACT).
        """
        if   ang_num == 60:
            projGeom = self.projGeomLACT60
        elif ang_num == 90:
            projGeom = self.projGeomLACT90
        else:
            raise ValueError(f"Unsupported LACT angle: {ang_num}")

        # choose volume geometry
        if ang_num == 580:
            volGeom = self.volGeom512
        else:
            volGeom = self.volGeom

        rec_id  = astra.data2d.create('-vol',  volGeom)
        proj_id = astra.data2d.create('-sino', projGeom, proj)
        cfg     = astra.astra_dict('FBP_CUDA')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId']    = proj_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)
        rec = astra.data2d.get(rec_id)

        # cleanup
        astra.algorithm.delete(alg_id)
        astra.data2d.delete(rec_id)
        astra.data2d.delete(proj_id)

        return rec