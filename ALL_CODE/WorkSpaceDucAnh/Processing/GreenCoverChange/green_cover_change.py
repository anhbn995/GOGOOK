import os, glob
import rasterio
import numpy as np
from tqdm import tqdm

def find_change_2_mask_different(fp_change_out, fp_img_truoc, fp_img_sau, value_class_truoc=255, value_class_sau=255, value_mat=1, value_them=2):
    with rasterio.open(fp_img_truoc) as src1:
        mask_truoc = src1.read()
        meta = src1.meta

    with rasterio.open(fp_img_sau) as src2:
        mask_sau = src2.read()

    idx_truoc = np.where(mask_truoc == value_class_truoc)
    idx_sau = np.where(mask_sau == value_class_sau)

    mask_mat = np.zeros_like(mask_truoc, dtype='uint8')
    mask_mat[idx_truoc] = value_mat
    mask_mat[idx_sau] = 0

    mask_them = np.zeros_like(mask_truoc, dtype='uint8')
    mask_them[idx_sau] = value_them
    mask_them[idx_truoc] = 0

    mask_rs = mask_mat + mask_them
    print('giá trị change:', np.unique(mask_rs))
    # print(mask_rs.shape)
    # print(meta)
    with rasterio.open(fp_change_out, 'w', **meta) as dst:
        dst.write(mask_rs)
        # dst.update_tags(AREA_OR_POINT="green='1'")
        dst.write_colormap(1, {
                0: (0,0,0, 0), 
                1: (255,0,0,0),
                2: (31,255,15,0)
                })




def main1_find_change_in_1folder(dir_out, dir_in, value_class_truoc, value_class_sau, value_mat, value_them):
    """
        Chạy với các ảnh trong cùng 1 folder, các năm được cho hết vào 1 folder ảnh, và kích thước các ảnh là như nhau.
    """
    os.makedirs(dir_out, exist_ok=True)
    list_fp_in = glob.glob(os.path.join(dir_in,'*.tif'))
    cap_lien_ke = [*zip(list_fp_in, list_fp_in[1:])]
    for fp_img_truoc, fp_img_sau in tqdm(cap_lien_ke):
        fname_change = os.path.basename(fp_img_truoc).replace('.tif','_vs_') + os.path.basename(fp_img_sau)
        print(fname_change)
        fp_change_out = os.path.join(dir_out, fname_change)
        find_change_2_mask_different(fp_change_out, fp_img_truoc, fp_img_sau, value_class_truoc=value_class_truoc, value_class_sau=value_class_sau, value_mat=value_mat, value_them=value_them)


def main2_find_change_in_2folder(dir_out, dir_truoc, dir_sau, value_class_truoc, value_class_sau, value_mat, value_them):
    """
        Chạy với các ảnh trong 2 folder khác năm nhau, có thể là mỗi ảnh ở folder 2021 vs 2022
        tên các ảnh phải giống nhau, và các ảnh cùng cặp cùng kích thước.
    """

    """
        Cái này là các tên file ở 2 foler là giống nhau:
    """
    os.makedirs(dir_out, exist_ok=True)
    for fp in tqdm(glob.glob(os.path.join(dir_truoc, '*.tif')), desc='process'):
        fname = os.path.basename(fp)
        fp_img_truoc = os.path.join(dir_truoc, fname)
        fp_img_sau = os.path.join(dir_sau, fname)
        fp_change_out = os.path.join(dir_out, fname + '.tif')
        find_change_2_mask_different(fp_change_out, fp_img_truoc, fp_img_sau, value_class_truoc=value_class_truoc, value_class_sau=value_class_sau, value_mat=value_mat, value_them=value_them)


    """
        Cái này đã đk custom 1 chút vs tên file
    """
    # os.makedirs(dir_out, exist_ok=True)
    # for fp in tqdm(glob.glob(os.path.join(dir_truoc, '*.tif')), desc='process'):
    #     fname = os.path.basename(fp).replace('21_label.tif','')
    #     fp_img_truoc = os.path.join(dir_truoc, fname + '21_label.tif')
    #     fp_img_sau = os.path.join(dir_sau, fname + '22_label.tif')
    #     fp_change_out = os.path.join(dir_out, fname + '.tif')
    #     find_change_2_mask_different(fp_change_out, fp_img_truoc, fp_img_sau, value_class_truoc=value_class_truoc, value_class_sau=value_class_sau, value_mat=value_mat, value_them=value_them)


if __name__=='__main__':
    value_class_truoc = 1
    value_class_sau = 1
    value_mat = 1
    value_them = 2

    """Chạy với nhiều ảnh ở 2 folder ảnh khác nhau"""
    # dir_truoc = r'E:\TMP_XOA\Forest_tiep\2021_mask'
    # dir_sau = r'E:\TMP_XOA\Forest_tiep\2022_mask'
    # dir_out = r'E:\TMP_XOA\Forest_tiep\rs_change_raster'
    # main2_find_change_in_2folder(dir_out, dir_truoc, dir_sau, value_class_truoc, value_class_sau, value_mat, value_them)

    """Chạy với nhiều ảnh ở cùng 1 folder"""
    # dir_in = r'E:\TMP_XOA\GreenCoverChange\Melbourne_San Jose\SAN_JOSE\3_mask_only_green'
    # dir_out = r'E:\TMP_XOA\GreenCoverChange\Melbourne_San Jose\SAN_JOSE\4_change'
    # main1_find_change_in_1folder(dir_out, dir_in, value_class_truoc, value_class_sau, value_mat, value_them)

    """Chay voi dau vao la 2 anh truoc va sau"""
    fp_img_truoc = r'/home/skm/SKM16/Tmp/HIEEU/GreenChang/google_ressult.tif'
    fp_img_sau = r'/home/skm/SKM16/Tmp/HIEEU/GreenChang/ngp_5km_ressult.tif'
    fp_change_out = r'/home/skm/SKM16/Tmp/HIEEU/GreenChang/rs/rs_change.tif'
    find_change_2_mask_different(fp_change_out, fp_img_truoc, fp_img_sau, value_class_truoc=value_class_truoc, value_class_sau=value_class_sau, value_mat=value_mat, value_them=value_them)


    import rasterio
    from rasterio.features import shapes
    from shapely.geometry import shape
    from tqdm import tqdm
    from geopandas import GeoDataFrame
    
    # out_dir = os.path.join(dir_out, 'shp')
    dir_out = r'/home/skm/SKM16/Tmp/HIEEU/GreenChang/rs'
    out_dir_shape = r'/home/skm/SKM16/Tmp/HIEEU/GreenChang/rs_shp'

    os.makedirs(out_dir_shape, exist_ok=True)

    for fp in tqdm(glob.glob(os.path.join(dir_out,'*.tif'))):
        fname_shp = os.path.basename(fp).replace('.tif','.shp')
        
        with rasterio.open(fp) as src:
            data = src.read(1, masked=True)
            # print(data)
            # Use a generator instead of a list
            shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))
            # print(shape_gen)
            # or build a dict from unpacked shapes
            gdf = GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)
        gdf['class'] = gdf['class'].replace([1], 'Green to Non')
        gdf['class'] = gdf['class'].replace([2], 'Non to Green')
        
        gdf.to_file(os.path.join(out_dir_shape, fname_shp))