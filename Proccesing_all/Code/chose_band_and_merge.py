import glob
import os
import sys
from subprocess import call
import shutil

get_image_band = {
    "/media/skymap/SKM/ghep_band/test1.tif": [1,2],
    "/media/skymap/SKM/ghep_band/test2.tif": [1],

}
output_file = "/media/skymap/SKM/download1.tif"
# chu y chu y
tmp_dir = "/media/skymap/SKM/tmp"

# main
image_sort = [*get_image_band]
image_sort.sort()
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

for im_ in image_sort:
    value = get_image_band[im_]
    name_file = os.path.basename(im_)
    for idx, band in enumerate(value):
        out_tmp = tmp_dir + "/" + name_file[:-4] + str(idx) + "_" + str(band) + ".vrt" 
        string = 'gdal_translate -b {} "{}" "{}"'.format(band, im_, out_tmp)
        os.system(string)

band_merge = glob.glob(tmp_dir+"/*.vrt")
band_merge.sort()
list_string = ['gdal_merge.py','-of','gtiff','-o']
list_string.append(output_file)
list_string.append("-separate")
list_string.extend(band_merge)
call(list_string)

try:
    shutil.rmtree(tmp_dir)
except OSError as e:
    print("Error: %s : %s" % (tmp_dir, e.strerror))