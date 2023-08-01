import os, glob
from tqdm import tqdm
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
RAW_PROFILE = cog_profiles.get("deflate")


def _translate(src_path, dst_path, profile=RAW_PROFILE, profile_options=None, **options):
    """
        Convert image to COG.
        NOTE : nó sẽ thay đổi luôn file gốc nếu   src_path == dst_path.
    """
    # Format creation option (see gdalwarp `-co` option)
    if profile_options is None:
        profile_options = {}
    output_profile = profile
    output_profile.update(dict(BIGTIFF="IF_SAFER"))
    output_profile.update(profile_options)

    # Dataset Open option (see gdalwarp `-oo` option)
    config = dict(
        GDAL_NUM_THREADS="ALL_CPUS",
        GDAL_TIFF_INTERNAL_MASK=True,
        GDAL_TIFF_OVR_BLOCKSIZE="128",
    )

    cog_translate(
        src_path,
        dst_path,
        output_profile,
        config=config,
        in_memory=False,
        quiet=False,
        **options,
    )
    return True


if __name__=="__main__":


# Chay nhiều ảnh
    # import sys
    # sys.path.append(r'E:\WorkSpaceDucAnh')
    # fp_txt_runed = r'E:\WorkSpaceDucAnh\Tmp\txt_runed.txt'
    # from ultils import read_txt
    # list_runed = read_txt(fp_txt_runed)

    # dir_imgA = r'E:\WorkSpaceSkyMap\Change_detection_Dubai\Data_Project\img2021_2022_unstack\A'
    # dir_imgB = r'E:\WorkSpaceSkyMap\Change_detection_Dubai\Data_Project\img2021_2022_unstack\B'

    # list_fname = [os.path.basename(fp) for fp in glob.glob(os.path.join(dir_imgA, '*.tif'))]
    # print(list_runed)
    # list_fname = list(set(list_fname).difference(set(list_runed)))
    # print(list_fname)
    # for fname in tqdm(list_fname, desc='Cog ALL: '):
    #     _translate(os.path.join(dir_imgA, fname), os.path.join(dir_imgA, fname))
    #     _translate(os.path.join(dir_imgB, fname), os.path.join(dir_imgB, fname))
    #     with open(fp_txt_runed, "a+") as file_object:
    #         file_object.seek(0)
    #         data = file_object.read(100)
    #         if len(data) > 0 :
    #             file_object.write("\n")
    #         file_object.write(fname)

## Chay một ảnh 
    for fp_img in [r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/img_ori/2023-06_mosaic.tif']:
    ## Sửa chính ảnh đó
        # fp_img = r"/home/skm/SKM16/Data/ThaiLandChangeDetection/Building_change_stanet/image8band_unstack_rgb (copy)/B/stack.tif"
        # _translate(fp_img, fp_img)
        

    # # tạo file khác thì
        fp_img_rs = fp_img.replace('.tif','_cog.tif')
        _translate(fp_img, fp_img_rs)
        
## Chay một ảnh 
    # list_name_dir = ["AOI_1","AOI_2","AOI_3","AOI_7","AOI_8","AOI_9","AOI_10","AOI_11","AOI_12","AOI_14","AOI_15"]
    # dir_img = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Regis_4band_original/Image_origin"
    # dir_cog_out = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Regis_4band_original/Image_origin_cog"
    
    # for fname in list_name_dir:
    #     dir_img_AOI = os.path.join(dir_img, fname)
    #     dir_cog_out_AOI = os.path.join(dir_cog_out, fname)
    #     os.makedirs(dir_cog_out_AOI, exist_ok=True)
        
    #     for fp_img in glob.glob(os.path.join(dir_img_AOI, '*.tif')):
    #     ## Sửa chính ảnh đó
    #         # fp_img = r"/home/skm/SKM16/Data/ThaiLandChangeDetection/Building_change_stanet/image8band_unstack_rgb (copy)/B/stack.tif"
    #         # _translate(fp_img, fp_img)
            

    #     # # tạo file khác thì
    #         # fp_img_rs = fp_img.replace('.tif','_cog.tif')
    #         fp_img_rs = os.path.join(dir_cog_out_AOI, os.path.basename(fp_img))
    #         _translate(fp_img, fp_img_rs)