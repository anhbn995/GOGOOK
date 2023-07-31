import rasterio
import cv2 as cv
import numpy as np 
from rasterio.features import sieve

fp_img = r"/home/skm/SKM/WORK/Demo_Kuwait/Kuwait_Planet_project/Img/Result_v2/20220404_132910_ssc17_u0001_visual_clip.tif"
fp_out = r"/home/skm/SKM/WORK/Demo_Kuwait/Kuwait_Planet_project/Img/Result_v2/20220404_132910_ssc17_u0001_visual_clip_rm_ok5.tif"

src = rasterio.open(fp_img)
mask = src.read()[0]

kernel = np.ones((6,6),np.uint8)
erosion = cv.erode(mask, kernel, iterations = 1)

profile = src.profile
sieved_msk = sieve(erosion, size=3000)

dilation = cv.dilate(sieved_msk,kernel,iterations = 1)

with rasterio.open(fp_out, 'w', **profile) as dst:
    dst.write(np.array([dilation]))


