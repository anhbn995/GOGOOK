import fiona
import rasterio.mask
import geopandas as gpd
import os, glob

def get_list_name_file(path_folder, name_file = '*.tif'):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        _, tail = os.path.split(file_)
        list_img_dir.append(tail)
    return list_img_dir

def crop_by_AOI(fp_shp, fp_img, fp_out_img_cut):

    # with fiona.open(fp_shp, "r") as shapefile:
    #     shapes = [feature["geometry"] for feature in shapefile]

    shapes = gpd.read_file(fp_shp)
    shapes = shapes.to_crs("EPSG:4326")
    shapes = shapes.to_json()
    import json
    shapes = json.loads(shapes)
    shapes = [shape["geometry"] for shape in shapes["features"]]


    with rasterio.open(fp_img) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform})

    with rasterio.open(fp_out_img_cut, "w", **out_meta) as dest:
        dest.write(out_image)

fp_shp = r"C:\Users\SkyMap\Desktop\b32648_a.shp"
# fp_shp = r"C:\Users\SkyMap\Desktop\a.shp"
# fd_img = r"Z:\DA\\2_GreenSpaceSing\Sentinel1_.Sar\data1_scene"
# fd_img = r"Z:\DA\2_GreenSpaceSing\T1_T9_sence"
fd_img = r"Z:\DA\2_GreenSpaceSing\xoa"
fd_img_out_cut = r"Z:\DA\\2_GreenSpaceSing\xoaaaaaaaaaSen2xoa4326_crop"
if not os.path.exists(fd_img_out_cut):
    os.makedirs(fd_img_out_cut)

# list_fname = get_list_name_file(fd_img)
# for name in list_fname:
#     fp_img = os.path.join(fd_img, name)
#     fp_out_img_cut = os.path.join(fd_img_out_cut, name)
#     crop_by_AOI(fp_shp, fp_img, fp_out_img_cut)


for i in range(1,10):
    if i == 0:
        continue
    else:
        name = f"T{i}"
        dir_img = os.path.join(fd_img, name)
        dir_crop_out = os.path.join(fd_img_out_cut, name)
        if not os.path.exists(dir_crop_out):
            os.makedirs(dir_crop_out)
            
        list_fname = get_list_name_file(dir_img)
        for name in list_fname:
            fp_img = os.path.join(dir_img, name)
            fp_out_img_cut = os.path.join(dir_crop_out, name)
            crop_by_AOI(fp_shp, fp_img, fp_out_img_cut)
            


# fp_img = r"Z:\DA\2_GreenSpaceSing\T1_T9_sence\T1\S2B_MSIL1C_20210115T032059_N0209_R118_T48NUG_20210115T053909.tif"
# fp_out_img_cut = r"Z:\DA\2_GreenSpaceSing\NguyenDucAnhXoa\a_32648.tif"                
# crop_by_AOI(fp_shp, fp_img, fp_out_img_cut)