from osgeo import gdal, gdalconst, ogr, osr
import numpy as np


def cal_bancut(image_path,num_channel):
    dataset = gdal.Open(image_path)
    band_cut_th = {k: dict(max=0, min=0) for k in range(num_channel)}
    for i_chan in range(num_channel):
        values_ = dataset.GetRasterBand(i_chan+1).ReadAsArray().astype(np.float16)
        values_[values_==0] = np.nan
        band_cut_th[i_chan]['max'] = np.nanpercentile(values_, 98)
        band_cut_th[i_chan]['min'] = np.nanpercentile(values_, 2)
        # print(band_cut_th[i_chan]['max'])
        # print(band_cut_th[i_chan]['min'])
    return band_cut_th


def percentile(raster_img, min_max_percentiles_all_band):
    
    img_float_01[i] = np.interp(raster_img[i], (min_tmp, max_tmp), (0, 1))



image_path = r"/home/skm/SKM16/Work/Npark_planet/image_roi_rac/20220313_023541_10_2465_3B_AnalyticMS_SR_8b_clip.tif"
num_channel = 8
min_max_bandi = cal_bancut(image_path,num_channel)
print(min_max_bandi)