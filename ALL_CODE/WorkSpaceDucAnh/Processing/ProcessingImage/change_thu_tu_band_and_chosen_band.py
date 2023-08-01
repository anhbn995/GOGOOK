import glob, os
import rasterio
import numpy as np

def change_idx_band(fp_out, fp_img_in, list_idx_band_change_and_chose):
    """Thay doi thu tu band

    Args:
        fp_out (str): duong dan anh dau ra cua anh sau khi thay doi thu tu band
        fp_img_in (str): duong dan anh dau vao, anh muon thay doi thu tu band
        list_idx_band_change_and_chose (list):  list indx cua thu tu band moi vd muoon chuyen anh ban dau voi band 3 len dau  
                                                va anh co 4 band thi list_idx_band_change_and_chose = [3,2,1,4] (index o day da cong them 1)
                                                va co the chon it band hon so luong bang dang co.
    """
    
    os.makedirs(os.path.dirname(fp_out), exist_ok=True)
    with rasterio.open(fp_img_in) as src:
        img = src.read()
        meta = src.meta
        
    numbands = len(list_idx_band_change_and_chose)
    list_idx_band_change_and_chose = (np.array(list_idx_band_change_and_chose) - 1).tolist()
    
    img_new = img[list_idx_band_change_and_chose, :]
    meta.update({
                    'count':numbands
                })
    
    with rasterio.open(fp_out, 'w', **meta) as dst:
        dst.write(img_new)
    print(f"Done: {os.path.basename(fp_img_in)}")
    
if __name__=="__main__":
    list_name_dir = ["AOI_1","AOI_2","AOI_3","AOI_7","AOI_8","AOI_9","AOI_10","AOI_11","AOI_12","AOI_14","AOI_15"]
    for name_dir in list_name_dir:
        in_dir = os.path.join('/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Img_origin', name_dir)
        out_dir = os.path.join('/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Img_origin_4band_BGRNir', name_dir)
        list_idx_band_change_and_chose = [2, 4, 6, 8] # so bang bat dau tu 1
        
        os.makedirs(out_dir, exist_ok=True)
        for fp in glob.glob(os.path.join(in_dir,'*.tif')):
            fn = os.path.basename(fp)
            fp_img_in = os.path.join(in_dir, fn)
            fp_out = os.path.join(out_dir, fn)
            change_idx_band(fp_out, fp_img_in, list_idx_band_change_and_chose)   
        
        
    """Chuan"""
    # in_dir = r'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/img/img_original_8band'
    # out_dir = r"/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/img/img_original_8band/Img_original_2468_BGRNir"
    # list_idx_band_change_and_chose = [2, 4, 6, 8] # so bang bat dau tu 1
    
    # os.makedirs(out_dir, exist_ok=True)
    # for fp in glob.glob(os.path.join(in_dir,'*.tif')):
    #     fn = os.path.basename(fp)
    #     fp_img_in = os.path.join(in_dir, fn)
    #     fp_out = os.path.join(out_dir, fn)
    #     change_idx_band(fp_out, fp_img_in, list_idx_band_change_and_chose)   