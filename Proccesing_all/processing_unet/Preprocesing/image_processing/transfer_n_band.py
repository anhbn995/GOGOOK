import cv2
import numpy as np
import gdal
import sys
import os
import time
image_source = os.path.abspath(sys.argv[1])
image_taget = os.path.abspath(sys.argv[2])
image_name = os.path.basename(image_taget)[:-4]
num_channel = int((sys.argv[3]))
parent = os.path.dirname(image_taget)


def trainsfer_image(data_source,data_target,num_channel):
    result =[]
    for i_chain in range(num_channel):
        chain_source = data_source[i_chain]
        chain_target = data_target[i_chain]
        chain_result = trainfer_chain(chain_source,chain_target)
        result.append(chain_result)
    return np.asarray(result)
def trainfer_chain(chain_source,chain_target):
    source_mean,source_std = image_stats(chain_source)
    target_mean,target_std = image_stats(chain_target)
    chain_result = chain_target - target_mean
    chain_result = (target_std / source_std) * chain_result
    chain_result = chain_result+source_mean
    return np.clip(chain_result, 0, 255)

def image_stats(chain):
    chain = chain.astype(np.float32)
    (chainMean, chainStd) = (chain.mean(), chain.std())
    return (chainMean, chainStd)

def main():
    ds_source = gdal.Open(image_source)
    data_source = ds_source.ReadAsArray()

    ds_target = gdal.Open(image_taget)
    data_target = ds_target.ReadAsArray()

    data_result = trainsfer_image(data_source,data_target,num_channel)
    output = os.path.join(parent,image_name+'_transfer.tif')
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(output,ds_target.RasterXSize,ds_target.RasterYSize,(data_result.shape[0]),gdal.GDT_Byte)#gdal.GDT_Byte/GDT_UInt16
    for i in range(1,data_result.shape[0]+1):
        dst_ds.GetRasterBand(i).WriteArray(data_result[i-1])
        dst_ds.GetRasterBand(i).ComputeStatistics(False)
    dst_ds.SetProjection(ds_target.GetProjection())
    dst_ds.SetGeoTransform(ds_target.GetGeoTransform())

if __name__=="__main__":
    x1 = time.time()
    main()
    print(time.time() - x1, "second")