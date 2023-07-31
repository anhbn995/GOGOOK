import os, glob
import rasterio
import numpy as np
from tqdm import tqdm

def sum_all(fp_img, fp_build_up, fp_road, fp_water, fp_out):
    list_all_class = [fp_build_up, fp_road, fp_water]

    with rasterio.open(fp_img) as src:
        mask_openland = src.read_masks(1)
        meta =src.meta
    meta.update({'count': 1,
                  'nodata': 0})
    # mask_openland = np.array([mask_openland])

    # print(mask_openland.shape)
    idx_nodata = np.where(mask_openland == 0)
    mask_openland[mask_openland != 0] = 0  
    for fp_class in list_all_class:
        with rasterio.open(fp_class) as src:
            mask_class = src.read()[0]
        mask_class[mask_class != 0] = 1
        mask_openland += mask_class

    mask_openland[mask_openland != 0] = 1
    mask_openland[idx_nodata] = 1
    with rasterio.open(fp_out, 'w', **meta) as dst:
        dst.write(np.array([mask_openland]))

def bo_file_tao_ra_muon_nhat_trong_list(list_fp):
    time_max = 0
    fp_break = 's'
    for fp in list_fp:
        time_create = os.path.getmtime(fp)
        if time_create > time_max:
            time_max = time_create
            fp_break = fp
    list_fp.remove(fp_break)
    return list_fp

def keep_list_fp_dont_have_list_eliminate(list_have_all, list_eliminate):
    list_eliminate = [os.path.basename(fp) for fp in list_eliminate]
    if list_eliminate:
        list_keep = []
        for fp in list_have_all:
            if os.path.basename(fp) not in list_eliminate:
                list_keep.append(fp)
        return list_keep
    else:
        return list_have_all


def main(dir_img, dir_buildUp, dir_road, dir_water, dir_out):
    list_img = glob.glob(os.path.join(dir_img, '*.tif'))
    
    list_runed = glob.glob(os.path.join(dir_out,'*.tif'))
    list_runed = bo_file_tao_ra_muon_nhat_trong_list(list_runed)
    list_img = keep_list_fp_dont_have_list_eliminate(list_img, list_runed)
    print(len(list_runed), 'da chay')
    print(len(list_img), 'tru')
    
    for fp_img in tqdm(list_img, desc='Run sum class: '):
        name_f = os.path.basename(fp_img)
        # print(name_f)
        fp_build_up = os.path.join(dir_buildUp, name_f)
        fp_road = os.path.join(dir_road, name_f)
        fp_water = os.path.join(dir_water, name_f)
        fp_out = os.path.join(dir_out, name_f)
        sum_all(fp_img, fp_build_up, fp_road, fp_water, fp_out)
    print('DONE')
    
if __name__ == '__main__':
    # Input
    dir_img = r"/home/skm/SKM16/Work/OpenLand/all_tif/"
    dir_buildUp = r"/home/skm/SKM16/Work/OpenLand/Result_Final/Buildup"
    dir_road = r"/home/skm/SKM16/Work/OpenLand/Result_Final/Road"
    dir_water = r"/home/skm/SKM16/Work/OpenLand/Result_Final/Water"

    # Output
    dir_out = r"/home/skm/SKM16/Work/OpenLand/Result_Final/zzzzzzzzzzzzz"
    os.makedirs(dir_out, exist_ok=True)
    #Run
    main(dir_img, dir_buildUp, dir_road, dir_water, dir_out)
    