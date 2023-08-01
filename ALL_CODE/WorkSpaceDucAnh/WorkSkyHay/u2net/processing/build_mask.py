import os, glob
import rasterio
import numpy as np
import geopandas as gpd
import rasterio.features
from tqdm import tqdm

def create_mask_with_class(out_path_mask, img_path, shp_path, gen_all = True, value_all = 255, gen_class_unique = False, list_class_choosen = None, property_name = None):
    """
        - có thể gen toàn bộ shape -> mask: chỉ cần gen_all = True và chọn giá trị all pixel, còn lại k cần truyền (o care).
        - có thể gen mỗi 1 loại polygon -> mỗi loại 1 giá trị mask: khi đó phải quan tâm các giá trị sau value_all.
        - có thể chọn một số loại class gộp -> thành 1 giá trị mask: khi đó mới cần quan tâm gen_class_unique
    """
    # CHECK MASK ZEROS
    check_mask_zeros = False

    # ĐỌC ẢNH VÀ UPDATE META MASK
    with rasterio.open(img_path) as src:
        meta = src.meta
        height, width = src.height, src.width
        tr = src.transform
        crs_img = src.crs

    meta.update({   
        'count': 1, 
        'nodata': 0,
        'dtype': 'uint8'
        })


    df_shape = gpd.read_file(shp_path)
    if df_shape.crs.to_string() != crs_img.to_string():
        df_shape = df_shape.to_crs(epsg=str(crs_img.to_epsg()))
    
    # df_shape = df_shape[df_shape.is_valid]
    # XỬ LÝ GEN TẤT CẢ HAY LÀ GEN THEO CLASS
    if gen_all:
        print(f"\nTạo tất cả polygon về 1 mask vs giá trị {value_all} ...")
        # print(df_shape)
        shapes = df_shape['geometry']
        if not len(df_shape):
            print("Shape file này là rỗng!")
            check_mask_zeros = True
    else:
        print("\nTạo mỗi polygon thành 1 giá trị khác nhau của mask ...")
        df_shape['valu'] = 0
        # check so luong class
        try:
            list_object_in_field = np.unique(df_shape[property_name])
        except:
            print(f"Không có trường '{property_name}'")
            return print('Error !!!')
        print(f'check {list_class_choosen} intersect with {list_object_in_field}')
        if set(list_class_choosen)&set(list_object_in_field):
            i = 0
            for class_name in list_class_choosen:
                i+=1
                print(f"- lớp {class_name} giá trị là {i}")
                if class_name in list_object_in_field:
                    df_shape.loc[df_shape[property_name] == class_name, 'valu']= i
                else:
                    continue
            shapes = df_shape[['geometry', 'valu']]
            shapes = list(map(tuple, shapes.values))
        else:
            print('roi vao exception mask trắng')
            print(f"{list_object_in_field} không chứa {list_class_choosen}")
            check_mask_zeros = True

    # TẠO MASK
    if check_mask_zeros:
        mask = np.zeros((height, width), dtype='uint8')
    else:
        mask = rasterio.features.rasterize(shapes, out_shape=(height, width), transform=tr)
    if gen_all or gen_class_unique:
        print(f'\nGộp tất cả class thành giá trị {value_all}')
        mask[mask!=0] = 1
        mask = mask*value_all
    with rasterio.open(out_path_mask, 'w', **meta) as dst:
            dst.write(np.array([mask]))
    print(f'\nDone {os.path.basename(img_path)}')


def chay_nhieu_file_same_name(out_dir_mask, dir_img, shp_dir, gen_all = True, value_all = 255, gen_class_unique = False, list_class_choosen = None, property_name = None):
    for shp_path in glob.glob(os.path.join(shp_dir, '*.shp')):
        fname = os.path.basename(shp_path)
        img_path = os.path.join(dir_img, fname.replace('.shp','.tif'))
        out_path_mask = os.path.join(out_dir_mask, fname.replace('.shp','.tif'))
        create_mask_with_class(out_path_mask, img_path, shp_path, gen_all = gen_all, value_all = value_all, gen_class_unique = gen_class_unique, list_class_choosen = list_class_choosen, property_name = property_name)

def main(out_dir_mask, img_dir, shp_dir, gen_all = True, value_all = 255, gen_class_unique = False, list_class_choosen = None, property_name = None):
    # custom theo từng bài toán
    # Cái này cho bài toán các ảnh tif A1.tif, A2.tif, A3.tif .... có chung 1 shp là A.shp
    
    list_fp_img = glob.glob(os.path.join(img_dir, '*.tif'))
    for fp_img in tqdm(list_fp_img, desc='Genmask ... '):
        fname_img = os.path.basename(fp_img)
        # fname_shp = fname_img[:fname_img.index('_')]
        fname_shp = fname_img.replace('.tif', '')
        fp_shp = os.path.join(shp_dir, fname_shp + '.shp')
        print(fp_shp)
        out_fp_mask = os.path.join(out_dir_mask, os.path.basename(fp_img))
        print(out_fp_mask)
        create_mask_with_class(out_fp_mask, fp_img, fp_shp, gen_all = gen_all, value_all = value_all, gen_class_unique = gen_class_unique, list_class_choosen = list_class_choosen, property_name = property_name)


def custom_sau_buoc_step1(out_dir_mask, img_dir, shp_dir, gen_all = True, value_all = 255, gen_class_unique = False, list_class_choosen = None, property_name = None):
    import re
    pattern = r"(.*)(_cut_\d+\.tif)"
    
    list_fp_img = glob.glob(os.path.join(img_dir, '*.tif'))
    for fp_img in tqdm(list_fp_img, desc='Genmask ... '):
        fname_img = os.path.basename(fp_img)
        match = re.match(pattern, fname_img)
        if match:
            # fname_shp = fname_img[:fname_img.index('_')]
            fname_shp =  match.group(1)
            fp_shp = os.path.join(shp_dir, fname_shp + '.shp')
            print(fp_shp)
            out_fp_mask = os.path.join(out_dir_mask, os.path.basename(fp_img))
            print(out_fp_mask)
            create_mask_with_class(out_fp_mask, fp_img, fp_shp, gen_all = gen_all, value_all = value_all, gen_class_unique = gen_class_unique, list_class_choosen = list_class_choosen, property_name = property_name)


def nhieu_file_chung_1_label(out_dir_mask, img_dir, fp_shp, gen_all = True, value_all = 255, gen_class_unique = False, list_class_choosen = None, property_name = None):
    print(fp_shp)
    list_fp_img = glob.glob(os.path.join(img_dir, '*.tif'))
    for fp_img in tqdm(list_fp_img, desc='Genmask ... '):
        out_fp_mask = os.path.join(out_dir_mask, os.path.basename(fp_img))
        create_mask_with_class(out_fp_mask, fp_img, fp_shp, gen_all = gen_all, value_all = value_all, gen_class_unique = gen_class_unique, list_class_choosen = list_class_choosen, property_name = property_name)



if __name__ =='__main__':
    # img_path = r"E:\WorkSpaceSkyMap\Change_detection_SuperView\Data_process\Vegetation\img_stack_cutbox_veget\img_stack_0.tif"
    # shp_path = r"E:\WorkSpaceSkyMap\Change_detection_SuperView\Data_origin\label_vegettable\vegetation_change.shp"
    # # shp_path = r"C:\Users\SkyMap\Music\xoa.shp"
    # out_dir = r"E:\WorkSpaceSkyMap\Change_detection_SuperView\Data_process\Vegetation\img_stack_cutbox_veget_unstack\label"
    # os.makedirs(out_dir, exist_ok=True)
    # out_path_mask = os.path.join(out_dir, os.path.basename(img_path))#.replace('.shp','.tif'))

    # # Gen tat cả thành 1
    # gen_all = True
    
    # # Giá trị gen khi all hoac unique
    # value_all = 255

    # # Gộp các lớp thành 1
    # gen_class_unique = True

    # # Trường chứa class
    # property_name = 'Chng_Type'

    # #Class quan tâm
    # list_class_choosen = ['Building Demolition', 'New Building', 'Rooftop Change', 'Existing Building Extension']

    # create_mask_with_class(out_path_mask, img_path, shp_path, gen_all = gen_all, value_all = value_all, gen_class_unique = gen_class_unique, list_class_choosen = list_class_choosen, property_name = property_name)

    """Chay cho folder với hướng dẫn như main ten anh cung vs shp"""
    # dir_img = r"/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Water/img_unit8_cut_box_Water"
    # dir_shp = r"/home/skm/SKM16/A_CAOHOC/ALL_DATA/label/Label/Nuoc"
    # out_dir_mask = r"/home/skm/SKM16/A_CAOHOC/Build_data_train_uint8_cua_3Class/Water/img_unit8_cut_box_Water_mask"
    # os.makedirs(out_dir_mask, exist_ok=True)



    """Chay cho sau step1 voi ten cua shp co trong ten anh"""
    # dir_img = r"/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/Training_dataset/Img_original_2468_BGRNir_cut"
    # dir_shp = r"/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/label"
    # out_dir_mask = r"/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/Training_dataset/Img_original_2468_BGRNir_cut_mask"
    # os.makedirs(out_dir_mask, exist_ok=True)


    # dir_img = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Water/img_cut"
    # dir_shp = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/b_LabelBox_each_class_and_area/Water/A/label"
    # out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Water/img_cut_mask"
    # os.makedirs(out_dir_mask, exist_ok=True)
    
    # dir_img = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Water/A"
    # dir_shp = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/b_LabelBox_each_class_and_area/Water/A/label"
    # out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Water/A_mask"
    # os.makedirs(out_dir_mask, exist_ok=True)


    dir_img = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Green/img_ori_cut_V2"
    dir_shp = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/b_LabelBox_each_class_and_area/Green/B/LabelV2"
    out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Green/img_ori_cut_V2_mask"
    os.makedirs(out_dir_mask, exist_ok=True)
    
    dir_img = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/a_img_original/B"
    dir_shp = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Vector_fix/fix_green"
    out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Vector_fix/fix_green"
    os.makedirs(out_dir_mask, exist_ok=True)


    dir_img = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/a_img_original/B"
    dir_shp = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Vector_fix/fix_water"
    out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Vector_fix/fix_water"
    os.makedirs(out_dir_mask, exist_ok=True)
    
    dir_img = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/a_img_original/B"
    dir_shp = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Vector_fix/fix_mo"
    out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Vector_fix/fix_mo"
    os.makedirs(out_dir_mask, exist_ok=True)  
    
    dir_img = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Img_origin/AOI_1"
    dir_shp = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/data_cloud_planet/AOI_1"
    out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Rs_cloud/AOI_1_mask"
    os.makedirs(out_dir_mask, exist_ok=True)  
    
    dir_img = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Img_origin/AOI_12"
    dir_shp = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Fix_RS_Cach2/Shp/AOI_12"
    out_dir_mask = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Fix_RS_Cach2/mask/AOI_12_mask"
    os.makedirs(out_dir_mask, exist_ok=True)
    
    # Gen tat cả thành 1 gia tri
    gen_all = True
    
    # Giá trị gen khi all hoac unique
    value_all = 255

    # Gộp các lớp thành 1
    gen_class_unique = True

    # Trường chứa class
    property_name = 'class_name'

    #Class quan tâm
    list_class_choosen = ['Building Demolition', 'New Building', 'Rooftop Change', 'Existing Building Extension']

    """chay_nhieu_file_same_name"""
    # chay_nhieu_file_same_name(out_dir_mask, dir_img, dir_shp, gen_all = gen_all, value_all = value_all, gen_class_unique = False, list_class_choosen = None, property_name = None)
    
    
    # custom_sau_buoc_step1
    # for img_path in glob.glob(os.path.join(dir_img, '*.tif')):
    #     fname = os.path.basename(img_path)
    #     shp_path = os.path.join(dir_shp, fname.replace('.tif','.shp'))
    #     out_path_mask = os.path.join(out_dir_mask, fname)
    #     create_mask_with_class(out_path_mask, img_path, shp_path, gen_all = gen_all, value_all = value_all, gen_class_unique = gen_class_unique, list_class_choosen = list_class_choosen, property_name = property_name)


    dir_img = r"/home/skm/SKM16/Data/IIIII/Data_Train_Pond_fix/tmp/Data_oke_cut_img"
    dir_shp = r"/home/skm/SKM16/Data/IIIII/label/Pond/pond_add_image7/Pond.shp"
    out_dir_mask = r"/home/skm/SKM16/Data/IIIII/Data_Train_Pond_fix/tmp/Data_oke_cut_img_mask"
    os.makedirs(out_dir_mask, exist_ok=True) 
    nhieu_file_chung_1_label(out_dir_mask, dir_img, dir_shp, gen_all = gen_all, value_all = value_all, gen_class_unique = False, list_class_choosen = None, property_name = None)
#buoc truoc nen dung file "/home/skm/SKM/WORK/ALL_CODE/WorkSpaceDucAnh/Processing/ProcessingImage/step1_cut_img_by_box.py"