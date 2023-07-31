# -*- coding: utf-8 -*-
import glob, os
from osgeo import gdal, gdalconst, ogr, osr


def resize(image_path,path_create, resolution_origin, resolution_destination):
    image_id =  os.path.basename(image_path)
    # output = os.path.join(path_create,image_id)
    output=path_create
    ds = gdal.Open(image_path)
    # resolution_origin = 0.1
    # resolution_destination = 0.2
    size_destination = resolution_origin/resolution_destination*100
    print(resolution_destination/resolution_origin)
    options_list = [
    f'-outsize {size_destination}% {size_destination}%',
    '-of GTiff',
    '-r cubic',
    '-ot Byte'
    ] 
    options_string = " ".join(options_list)

    gdal.Translate(output,
                image_path,
                options=options_string)
    return output

"""Queensland"""
# image_path=r"/home/skm/SKM/WORK/USA/Queensland Mosaics/building_28354/img_28534/Mt_Isa_2017_10cm_Mosaic.tif"
# path_create=r"/home/skm/SKM/WORK/USA/Queensland Mosaics/building_28354/img_28534/Mt_Isa_2017_10cm_Mosaic_resize02.tif"
# resolution_origin = 0.1
# resolution_destination = 0.2

# image_path=r"/home/skm/SKM/WORK/USA/Queensland Mosaics/img_ori/Capricorn_Wide_Bay_2017_20cm_Mosaic.tif"
# path_create=r"/home/skm/SKM/WORK/USA/Queensland Mosaics/img_ori/Capricorn_Wide_Bay_2017_20cm_Mosaic_Result/Capricorn_Wide_Bay_2017_20cm_Mosaic_resize02.tif"
# resolution_origin = 0.2
# resolution_destination = 0.3

# image_path=r"/home/skm/SKM_OLD/public/DA/1_TreeCounting_All_DATA/Data_origin/create_data_train_one_type/img/Parrot_Sequioa_Ortho_V2_train.tif"
# path_create=r"/home/skm/SKM_OLD/public/DA/1_TreeCounting_All_DATA/Data_origin/create_data_train_one_type/img/resize1/Parrot_Sequioa_Ortho_V2_train.tif"
# resolution_origin = 0.023
# resolution_destination = 0.2

image_path=r"/home/skm/SKM_OLD/public/khalifacity-2020.tif"
path_create=r"/home/skm/SKM_OLD/public/khalifacity-2020_resize.tif"
resolution_origin = 0.45
resolution_destination = 0.3
# main
resize(image_path,path_create, resolution_origin, resolution_destination)