import glob
import os
import sys
from subprocess import call
import shutil
import gdal
from osgeo import gdal
# de nguy khong sua nhe
tmp_dir = "/home/skm/SKM_OLD/public/DA/2_GreenSpaceSing/Training_Stack_6band/Data/Stack/tmp"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

get_image_band = {
    "/home/skm/SKM_OLD/public/DA/2_GreenSpaceSing/Training_Stack_6band/Data/Sen2_cut/S1A_IW_GRDH_1SDV_20210219T224801_20210219T224826_036665_044EF6_F0DD_0.tif": [1,2,3,4],
    "/home/skm/SKM_OLD/public/DA/2_GreenSpaceSing/Training_Stack_6band/Data/sen1_crop/S1A_IW_GRDH_1SDV_20210219T224801_20210219T224826_036665_044EF6_F0DD_0.tif": [1,2],
}
output_file = "/home/skm/SKM_OLD/public/DA/2_GreenSpaceSing/Training_Stack_6band/Data/Stack/aaaa_khong.tif"

image_sort = [*get_image_band]

for im_ in image_sort:
    value = get_image_band[im_]
    name_file = os.path.basename(im_)
    for idx, band in enumerate(value):
        option = {'bandList':[band]}
        out_tmp = tmp_dir + "/" + name_file[:-4] + str(idx) + "_" + str(band) + ".vrt" 
        gdal.Translate(out_tmp, im_, **option)
        print(1)

band_merge = glob.glob(tmp_dir+"/*.vrt")
print(band_merge)
option_vrt = {"separate":"separate"}
gdal.BuildVRT(output_file, band_merge, **option_vrt)
try:
    shutil.rmtree(tmp_dir)
except OSError as e:
    print("Error: %s : %s" % (tmp_dir, e.strerror))