# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 11:16:55 2021

@author: SkyMap
"""

import rasterio 
import numpy as np


for i in range(0,5):
    fp_img = f"Z:\DA\\2_GreenSpaceSing\Training_Stack_6band\Data\Sen2_4326\S1A_IW_GRDH_1SDV_20210219T224801_20210219T224826_036665_044EF6_F0DD_{i}.tif"
    fp_mask = f"Z:\DA\\2_GreenSpaceSing\Training_Stack_6band\Data_training\Stack_mask\S1A_IW_GRDH_1SDV_20210219T224801_20210219T224826_036665_044EF6_F0DD_{i}.tif"
    out = f"Z:\DA\\2_GreenSpaceSing\Training_Stack_6band\Data_training\mask_norm\S1A_IW_GRDH_1SDV_20210219T224801_20210219T224826_036665_044EF6_F0DD_{i}.tif"
    src_img = rasterio.open(fp_img)
    src_mask = rasterio.open(fp_mask)
    
    img = src_img.read()
    mask = src_mask.read()
    
    mask_xoa = src_img.read_masks(1)
    mask_xoa = np.array([mask_xoa])
    ind_xoa = np.where(mask_xoa == 0)
    mask[ind_xoa] = 0
    profile = src_mask.profile
    with rasterio.open(out, 'w', **profile) as dst:
        dst.write(mask)
        
        
