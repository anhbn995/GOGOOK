import glob
import os
import sys
from subprocess import call
import rasterio
# input_name1 = r"/home/skm/SKM_OLD/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A  Dsc 12-February-2021.tif"
# input_name2 = r"/home/skm/SKM_OLD/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A Dsc 10-Oct-2021.tif"
# output_dir = r"/home/skm/SKM_OLD/public/changedetection_SAR/DA/Stack_"
# output_name = 'stack_Feb24_Feb'

input_name1 = r"/home/skm/SKM16/X/Test/img_origin (copy)/20181031.tif"
input_name2 = r"/home/skm/SKM16/X/Test/Landslide_Sentinel-2_DAnh/DaBac/Slope_10m.tif"
output_dir = r"/home/skm/SKM16/X/Landslide_Sentinel-2_DAnh/Training_ver_5band_goc/All_img_stack_big/img_big_test_stack"
output_name = '20181031_stack8'

# with rasterio.open(input_name2) as src:
#     meta = src.meta
# meta.update({'count':4})

# with rasterio.open(input_name1, 'r+', **meta) as src:
    # pass

if __name__ == "__main__":
    with rasterio.open(input_name2) as src:
        meta = src.meta
    meta.update({'count':4})

    with rasterio.open(input_name1, 'r+', **meta) as src:
        pass
        
    
    file_list = [input_name1, input_name2]
    with open(os.path.join(output_dir,'{}.txt'.format(output_name)), 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)
    print(file_list)
    list_string = ['gdal_merge.py','-of','gtiff','-o']
    output_file = os.path.join(output_dir,'{}.tif'.format(output_name))
    print(output_file)
    list_string.append(str(output_file))
    list_string.append("-separate")
    list_string.append("-ot")
    list_string.append("Float32")
    list_string.append("-tr")
    list_string.append("10")
    list_string.append("10")
    
    for file_name in file_list:
        list_string.append(file_name)
    call(list_string)