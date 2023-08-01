import os
import rasterio
from rasterio.merge import merge
import glob

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


def mosaic_gdal(dir_path, list_img_name, out_path):
    src_files_to_mosaic = []
    if not os.path.exists(out_path):
        for name_f in list_img_name:
            fp = os.path.join(dir_path, name_f)
            src = rasterio.open(fp)
            src_files_to_mosaic.append(src)
        mosaic, out_trans = merge(src_files_to_mosaic)
        write_image(mosaic, mosaic.shape[1], mosaic.shape[2], mosaic.shape[0], src.crs, out_trans, out_path)
    return out_path

if __name__=='__main__':
    # dir_path = r'E:\TMP_XOA\DuBai\Sentinel2base\img_sen2'
    # list_img_name = ['S2B_MSIL2A_20220523T064629_N0400_R020_T40RBN_20220523T095725.tif', 'S2A_MSIL2A_20220508T064631_N0400_R020_T40RDN_20220508T111311.tif', 'S2A_MSIL2A_20220518T064631_N0400_R020_T40RCP_20220518T124720.tif', 'S2A_MSIL2A_20220528T064631_N0400_R020_T40RCN_20220528T112412.tif']
    # out_path = r'E:\TMP_XOA\DuBai\Sentinel2base\ok.tif'
    # mosaic_gdal(dir_path, list_img_name, out_path)
    
    dir_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-04'
    list_img_name = [os.path.basename(fp) for fp in glob.glob(os.path.join(dir_path, '*.tif'))]
    
    # list_img_name = ['S2B_MSIL2A_20220523T064629_N0400_R020_T40RBN_20220523T095725.tif', 'S2A_MSIL2A_20220508T064631_N0400_R020_T40RDN_20220508T111311.tif', 'S2A_MSIL2A_20220518T064631_N0400_R020_T40RCP_20220518T124720.tif', 'S2A_MSIL2A_20220528T064631_N0400_R020_T40RCN_20220528T112412.tif']
    out_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-04/2023-04_mosaic.tif'
    mosaic_gdal(dir_path, list_img_name, out_path)