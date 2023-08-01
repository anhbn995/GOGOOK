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



data_dir = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8'
save_model_dir = r"/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8/model"
model_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/gen_cut128stride100_TruotLo_UINT8/model/gen_cut128stride100_TruotLo_UINT82.h5'
size_model = 128

name_model = os.path.basename(data_dir) + '_' + dt
log_tensorboard_dir = os.path.join(os.path.join(save_model_dir, name_model), "log_tensorboard")
os.makedirs(log_tensorboard_dir)
fp_out_model = os.path.join(os.path.join(save_model_dir, name_model), name_model + '.h5')
# fp_out_model = f'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/All_model_cloud/gen_Cloud_Uint8_4band_cut_512_stride_200_{dt}.h5'