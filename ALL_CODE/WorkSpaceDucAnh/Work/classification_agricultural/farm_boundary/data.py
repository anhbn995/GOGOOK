# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 09:39:14 2022

@author: AnhHo
"""

import rasterio
with rasterio.open(r"Z:\tmp_Nam\farm_boundary\test.tif") as src:
    image = src.read()
    src.bounds
from PIL import Image
im = Image.fromarray(image.transpose(1,2,0))
im.save(r"Z:\tmp_Nam\farm_boundary\test.jpg")