import os, glob
import rasterio

dir_mutil_class = "/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/RS_UNET_sai"
cmap_kgx = {
            1:(61, 212, 114),
            2:(255, 210, 58), 
            3:(57, 118, 210)
        }
for fp_out_union in glob.glob(os.path.join(dir_mutil_class,'*.tif')):
    with rasterio.open(fp_out_union) as src:
        img = src.read()
        
    img[img == 1] = 4
    img[img == 2] = 1
    img[img == 3] = 2
    img[img == 4] = 3 
    
    with rasterio.open(fp_out_union, 'r+') as dst:
        dst.write(img)
        dst.write_colormap(1, cmap_kgx)