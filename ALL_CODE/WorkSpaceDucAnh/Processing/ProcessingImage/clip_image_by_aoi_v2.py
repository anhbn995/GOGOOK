import rasterio
from rasterio.mask import mask
import json



import json
import os
import geopandas as gpd


# print(aoi)
# Mở file ảnh vệ tinh

# fp_need_cut = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Rs_tmp/Union/Rs_tmp/rs_box/2023-04_mosaic_cog_rs_oke.tif"
# fp_shp_cut = r"/home/skm/public_mount/tmp_ducanh/planet_basemap/adaro.geojson"
# out_dir = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Rs_tmp/Union/Rs_tmp/rs_box/clip_by_aoi/2023-04_mosaic_cog_rs_oke.tif"

fp_need_cut = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/RS_TEST_XOA/2023-06_mosaic_union_4326.tif'
fp_shp_cut = r'/home/skm/SKM16/Planet_GreenChange/aaaaaa.geojson'
out_dir = r"/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/RS_TEST_XOA/Clip/2023-06_mosaic_union.tif"

os.makedirs(out_dir, exist_ok=True)
with rasterio.open(fp_need_cut) as src:
    # Đọc nội dung của file GeoJSON chứa đối tượng AOI
    with open(fp_shp_cut) as f:
        aoi = json.load(f)
    # print(aoi)
    # aoi = gpd.read_file(r"/home/skm/public_mount/tmp_ducanh/planet_basemap/adaro.geojson")
    # Thực hiện cắt ảnh theo vùng quan tâm (AOI)
    out_image, out_transform = mask(src, aoi, crop=True)

    # Lưu ảnh cắt được ra file mới
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(out_image)
