import glob, os
import rasterio
import numpy as np
from tqdm import tqdm

def unique_multi_class_to_1_image(dict_fp_img, thu_tu_lop_va_gia_tri_lop, cmap, fp_out_union):
    # print(dict_fp_img)
    first_fp_class = dict_fp_img[thu_tu_lop_va_gia_tri_lop[0]]
    value_first = int(thu_tu_lop_va_gia_tri_lop[0][-1])

    with rasterio.open(first_fp_class) as src:
        img = src.read()
        meta = src.meta
    img[img!=0] = value_first
    try:
        for class_value in thu_tu_lop_va_gia_tri_lop[1:]:
            value_class = class_value[-1]
            fp_img_lop = dict_fp_img[class_value]

            with rasterio.open(fp_img_lop) as src:
                img_ = src.read()
                idex_khac0 = np.where(img_ != 0)
                del img_
                img[idex_khac0] = value_class
        print(fp_out_union)
        with rasterio.open(fp_out_union, 'w', **meta) as dst:
            dst.write(img)
            dst.write_colormap(1, cmap)
    except:
        print(first_fp_class, "ERROR")


def main_khong_gian_xanh(dir_grass, dir_tree, dir_water, thu_tu_lop_va_gia_tri_lop, cmap_kgx, dir_union):
    os.makedirs(dir_union, exist_ok=True)
    
    list_fn_img = [os.path.basename(fp) for fp in glob.glob(os.path.join(dir_grass,'*.tif'))]
    dict_fp_img = {}
    for fn_tif in tqdm(list_fn_img, desc="Tung anh"):
        fp_union_each_img = os.path.join(dir_union, fn_tif)
        dict_fp_img["fp_grass1"] = os.path.join(dir_grass, fn_tif)
        dict_fp_img["fp_tree2"] = os.path.join(dir_tree, fn_tif)
        dict_fp_img["fp_water3"] = os.path.join(dir_water, fn_tif)
        unique_multi_class_to_1_image(dict_fp_img, thu_tu_lop_va_gia_tri_lop, cmap_kgx, fp_union_each_img)

if __name__=="__main__":
    # dir_grass = r'/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/predict_CAOHOC_U2net_Grass'
    # dir_tree = r'/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/predict_CAOHOC_U2net_Tree'
    # dir_water = r'/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/predict_CAOHOC_U2net_Water'
    
    dir_grass = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Grass/img_unit8_cut_box_Grass_mask'
    dir_tree = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Tree/img_unit8_cut_box_Tree_mask'
    dir_water = r'/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Water/img_unit8_cut_box_Water_mask'
    
    dir_out_union = r"/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/Union_all_label"
    thu_tu_lop_va_gia_tri_lop = ["fp_grass1", 'fp_tree2', 'fp_water3']   # vao cuoi la o tren cung
    cmap_kgx = {
                1:(61, 212, 114),
                2:(255, 210, 58), 
                3:(57, 118, 210)
            }
    main_khong_gian_xanh(dir_grass, dir_tree, dir_water, thu_tu_lop_va_gia_tri_lop, cmap_kgx, dir_out_union)