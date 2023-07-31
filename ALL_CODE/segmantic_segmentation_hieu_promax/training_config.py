    # data_dir = r'/home/skm/SKM16/Tmp/HIEEU/training_dataset_dattrong'
    # model_path = ''
    # fp_out_model = r'/home/skm/SKM16/X/Data_MachineLearning/V2_sentinel/training_dataset/model/vacant_land.h5'
    # size_model = 256
    # training(data_dir, model_path, size_model, fp_out_model)
    
    # data_dir = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/gen_cut128stride100_TruotLo_UINT8'
    # model_path = ''
    # fp_out_model = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/gen_cut128stride100_TruotLo_UINT8/model/gen_cut128stride100_TruotLo_UINT82.h5'
    # size_model = 128
    # training(data_dir, model_path, size_model, fp_out_model)
    
    # data_dir = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8'
    # model_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8/model/gen_cut128stride100_TruotLo_UINT82.h5'
    # fp_out_model = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8/model/gen_cut128stride100_TruotLo_UINT82_them_vung_khong_totV2.h5'
    

    # data_dir = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/TrainingDataSet/Water/gen_Water_UINT8_4band_cut_512_stride_200'
    # model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948.h5'
    # fp_out_model = f'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_{dt}_V2_water.h5'
    # size_model = 512
    
    # data_dir = r'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/Training_dataset/V2/gen_Cloud_Uint8_4band_cut_512_stride_200'
    # model_path = r''
    # fp_out_model = f'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/All_model_cloud/gen_Cloud_Uint8_4band_cut_512_stride_200_{dt}.h5'
    # size_model = 512
    
    # data_dir = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/images/img_origin_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200'
    # model_path = r''
    # fp_out_model = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/images/img_origin_8bit_perimage/models/gen_Green_UINT8_4band_cut_512_stride_200.h5'
    # size_model = 512




import os
import datetime
now = datetime.datetime.now()
dt = now.strftime("%Y%m%d_%H%M%S")


"""Truot Lo"""
data_dir = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8'
save_model_dir = r"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard/Model_unet"
# model_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8/model/gen_cut128stride100_TruotLo_UINT82.h5'
model_path = ''
size_model = 128


# data_dir = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_Truot_Lo_Uint8_4band_cut_64_stride_32_time_20230514_213511'
# save_model_dir = r"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard"
# # model_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8/model/gen_cut128stride100_TruotLo_UINT82.h5'
# model_path = ''
# size_model = 64


# data_dir = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_Truot_Lo_Uint8_4band_cut_32_stride_30_time_20230514_225229'
# save_model_dir = r"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard"
# # model_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8/model/gen_cut128stride100_TruotLo_UINT82.h5'
# model_path = ''
# size_model = 32

"""Truot Lo"""






"""Planet miner"""
# data_dir = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/water_training/gen_Water_original_Original_4band_cut_512_stride_200_time_20230515_112131'
# save_model_dir = r"/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/a_model_all"
# model_path = ''
# size_model = 512


# data_dir = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/green_trainig/gen_Green_original_Original_4band_cut_512_stride_200_time_20230515_142534'
# save_model_dir = r"/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/a_model_all"
# model_path = ''
# size_model = 512


# """Water ALL BIG"""
class_IS = "Water"
data_dir = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Water/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750"
save_model_dir = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG"
# model_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/a_model_all/gen_Water_original_Original_4band_cut_512_stride_200_time_20230515_112131_20230515_162207v22/gen_Water_original_Original_4band_cut_512_stride_200_time_20230515_112131_20230515_162207v22.h5'
model_path = ""
data_type = "Uint16"
size_model = 512


class_IS = "Green"
data_dir = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/Green/gen_ALL_Green_Original_4band_cut_512_stride_200_time_20230518_100524"
save_model_dir = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG"
model_path = ""
data_type = "Uint16"
size_model = 512
"""Planet miner"""


if data_type.upper() == "UINT8":
    type_data = True
else:
    type_data = False
save_model_dir = os.path.join(save_model_dir, class_IS)
name_model = os.path.basename(data_dir) + '_' + dt + data_type
log_tensorboard_dir = os.path.join(os.path.join(save_model_dir, name_model), "log_tensorboard")
os.makedirs(log_tensorboard_dir)
fp_out_model = os.path.join(os.path.join(save_model_dir, name_model), name_model + '.h5')
# fp_out_model = f'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/All_model_cloud/gen_Cloud_Uint8_4band_cut_512_stride_200_{dt}.h5'