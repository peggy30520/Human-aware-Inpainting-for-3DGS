#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id,
        R,
        T,
        FoVx,
        FoVy,
        image,
        mask, # modify
        mask_bbd_list, # modify
        gt_alpha_mask,
        image_name,
        uid,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
    ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )

        # add
        self.mask = None
        if mask is not None:
            self.mask = mask.clamp(0.0, 1.0).to(self.data_device)

        self.mask_L = None
        self.mask_R = None
        self.mask_T = None
        self.mask_B = None
        self.bbd_amo = 0

        if mask_bbd_list is not None:
            self.bbd_amo = len(mask_bbd_list)
            mask_L = list(range(self.bbd_amo))
            mask_R = list(range(self.bbd_amo))
            mask_T = list(range(self.bbd_amo))
            mask_B = list(range(self.bbd_amo))
            
            for i, mask_bbd in enumerate(mask_bbd_list):
                mask_L[i] = mask_bbd[0]
                mask_R[i] = mask_bbd[1]
                mask_T[i] = mask_bbd[2]
                mask_B[i] = mask_bbd[3]
            
            self.mask_L = torch.tensor(mask_L, dtype=torch.int32, device='cuda')
            self.mask_R = torch.tensor(mask_R, dtype=torch.int32, device='cuda')
            self.mask_T = torch.tensor(mask_T, dtype=torch.int32, device='cuda')
            self.mask_B = torch.tensor(mask_B, dtype=torch.int32, device='cuda')
            
            
            #print("----mask_L_R_T_B----")
            #print(self.mask_L)
            #print(self.mask_R)
            #print(self.mask_T)
            #print(self.mask_B)
            

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .cuda()
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
