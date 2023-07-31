import os
import glob
import rasterio
import numpy as np

from osgeo import gdal
from rasterio.warp import reproject, Resampling

def write_image(data, height, width, numband, crs, tr, out):
    """
        Export numpy array to image by rasterio.
    """
    with rasterio.open(
                        out,
                        'w',
                        driver='GTiff',
                        height=height,
                        width=width,
                        count=numband,
                        dtype=data.dtype,
                        crs=crs,
                        transform=tr,
                        nodata=0,
                        ) as dst:
                            dst.write(data)

def reproject_image_goc(src_path, dst_path, dst_crs='EPSG:4326'):
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    temp_path = dst_path.replace('.tif', 'temp.tif')
    option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
    gdal.Translate(temp_path, src_path, options=option)
    option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
    gdal.Warp(dst_path, temp_path, options=option)
    os.remove(temp_path)
    return True

def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    if ds.crs.to_string() != dst_crs:
        print(f'convert to {dst_crs}')
        temp_path = dst_path.replace('.tif', 'temp.tif')
        option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
        gdal.Translate(temp_path, src_path, options=option)
        option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
        gdal.Warp(dst_path, temp_path, options=option)
        os.remove(temp_path)
    else:
        import shutil
        print(f'coppy image to {dst_path}')
        shutil.copyfile(src_path, dst_path)
        print('done coppy')
    return True

def check_base_img(box_path, base_path, use_box=True):
    print("Start check input file(box, image, base) and remove old mosaic tmp ...")
    if len(glob.glob(os.path.join(box_path, '*.shp')))==0 and use_box:
        raise Exception("Box folder is empty, please check %s"%(box_path))
    
    # for i in list_month:
    #     if len(glob.glob(os.path.join(i,'*.tif'))) == 0:
    #             raise Exception("Image folder is empty, please check %s"%(i))

    if glob.glob(os.path.join(base_path,'*.tif')):
        base_image = glob.glob(os.path.join(base_path,'*.tif'))[0]
    else:
        base_image = None
    return base_image

# def standard_coord(img_path, crs):
#     list_img = glob.glob(os.path.join(img_path, '*.tif'))
#     for i in list_img:
#         with rasterio.open(i) as dst:
#             crs_temp = dst.crs.to_string()
#         if crs_temp == crs:
#             pass
#         else:
#             reproject_image(i, i, dst_crs=crs)

def window_from_extent(xmin, xmax, ymin, ymax, aff):
        col_start, row_start = ~aff * (xmin, ymax)
        col_stop, row_stop = ~aff * (xmax, ymin)
        return ((int(row_start), int(row_stop)), (int(col_start), int(col_stop)))

def convert_profile(src_path, dst_path, out_path):
    kwargs = None
    _info = gdal.Info(dst_path, format='json')
    xmin, ymin = _info['cornerCoordinates']['lowerLeft']
    xmax, ymax = _info['cornerCoordinates']['upperRight']

    with rasterio.open(dst_path) as dst:
        dst_transform = dst.transform
        kwargs = dst.meta
        kwargs['transform'] = dst_transform
        dst_crs = dst.crs

    with rasterio.open(src_path) as src:
        window = window_from_extent(xmin, xmax, ymin, ymax, src.transform)
        src_transform = src.window_transform(window)
        data = src.read()

        with rasterio.open(out_path, 'w', **kwargs) as dst:
            for i, band in enumerate(data, 1):
                _band = src.read(i, window=window)
                # print(_band.shape)
                dest = np.zeros_like(_band) 
                reproject(
                    _band, dest,
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=src_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

                dst.write(dest, indexes=i)


if __name__=="__main__":
    src_path = '/home/quyet/DATA_ML/Projects/Green Cover Npark Singapore/results/T3-2022/T3-2022.tif'
    dst_path = '/home/quyet/DATA_ML/Projects/Green Cover Npark Singapore/results/T3-2022/DA/T3-2022_DA.tif'
    out_path = '/home/quyet/DATA_ML/Projects/Green Cover Npark Singapore/results/T3-2022/T3-2022_new.tif'
    convert_profile(src_path, dst_path, out_path)