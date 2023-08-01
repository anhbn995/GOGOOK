import os, glob
import rasterio

dir_mutil_class = "/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/RS_UNET"
cmap_kgx = {
            1:(61, 212, 114),
            2:(255, 210, 58), 
            3:(57, 118, 210)
        }
for fp_out_union in glob.glob(os.path.join(dir_mutil_class,'*.tif')):
    with rasterio.open(fp_out_union, 'r+') as dst:
        dst.write_colormap(1, cmap_kgx)