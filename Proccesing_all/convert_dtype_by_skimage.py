from skimage.util import img_as_ubyte
import rasterio

fp_img = r"/home/skm/SKM_OLD/public/TMP_DUCANH/aa/BaLat_S2B_20190516_G.tif"
out_fp = r"/home/skm/SKM_OLD/public/TMP_DUCANH/aa/BaLat_S2B_20190516_G_uint8.tif"
with rasterio.Env():
    with rasterio.open(fp_img) as src:
        img = src.read()
        profile = src.profile

    img_new = img_as_ubyte(img)
    print(profile)
    profile.update(dtype=rasterio.uint8)
    print(profile)
    with rasterio.open(out_fp, 'w', **profile) as dst:
        dst.write(img_new)
