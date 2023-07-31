from model import unet_basic
import tensorflow as tf
# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_test_xong_xoa.h5'#r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/Data_4band_Green/ImgRGBNir_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/Data_4band_Green/ImgRGBNir_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200_test_xong_xoa'#gen_Green_UINT8_4band_cut_512_stride_200_predict'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))



# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/planet/images_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/planet/images_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))

# /home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-03_8bit_perimage

model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050'
# size_model = 512
folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05'
folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/water/b'
size_model = 512
model = unet_basic((size_model, size_model, 4))


# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))


# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green.h5'
# # folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green'
# # folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8/all_result_mosaic/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/water/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green_ok'
size_model = 512
model = unet_basic((size_model, size_model, 4))


# gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water.h5
# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/water/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))



# CLOUD FREE PLANET
# model_path = r'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/All_model_cloud/gen_Cloud_Uint8_4band_cut_512_stride_200_20230505_094454.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_result_mosaic/gen_Cloud_Uint8_4band_cut_512_stride_200_20230505_094454'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))


# # model_path = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG/Water/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_232227Uint16/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_232227Uint16.h5'
# model_path = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG/Water/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_223218Uint16/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_223218Uint16.h5"

# model_path = r"/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG/Green/gen_ALL_Green_Original_4band_cut_512_stride_200_time_20230518_100524_20230518_100730Uint16/gen_ALL_Green_Original_4band_cut_512_stride_200_time_20230518_100524_20230518_100730Uint16.h5"
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/a_img_original/B'
# # folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Data_Test'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Rs_tmp/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_223218Uint16'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/Rs_tmp/gen_ALL_Green_Original_4band_cut_512_stride_200_time_20230518_100524_20230518_100730Uint16'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))




# # save_stuct_to_weight
# # model_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard/Model_unet/gen_cut128stride100_TruotLo_UINT8_20230515_232218/gen_cut128stride100_TruotLo_UINT8_20230515_232218.h5'
# model_path_struct = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/Model_v2_ThemVungKhongTot/all_model_tensorboard/Model_unet/gen_cut128stride100_TruotLo_UINT8_20230515_232218/gen_cut128stride100_TruotLo_UINT8_20230515_232218_luu_cau_truc2.h5'
# folder_image_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/bo_them_nodata/Img_uint8_crop/a'
# folder_output_path = r'/home/skm/SKM16/X/Lo/Full_Images_LandslideDetection_8bit_perimage/bo_them_nodata/Img_uint8_crop/oke11'
# size_model = 128
# # model = unet_basic((size_model, size_model, 4))
# # model.load_weights(model_path)
# # model.save(model_path_struct)
# model = tf.keras.models.load_model(model_path_struct)
