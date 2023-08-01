from model import unet_basic
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

model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050'
folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05'
folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/RS_gre'
size_model = 512
model = unet_basic((size_model, size_model, 4))


# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948.h5'
# # folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/RS_a'

# size_model = 512
# model = unet_basic((size_model, size_model, 4))

# """2_Indonesia_Mining_Exhibition_Data"""
# """uint8"""
# folder_image_path_ori = r'/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Img_Uint8_4band_BGRNir'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Rs_predict'
# folder_output_path = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Rs_predict"
# # folder_output_path_green = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Rs_predict"

# model_path_green = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green.h5'
# size_model = 512
# model_green = unet_basic((size_model, size_model, 4))

# model_path_water = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water.h5'
# size_model = 512
# model_water = unet_basic((size_model, size_model, 4))

# """Uint16"""
# folder_image_path_ori = r'/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Img_origin_4band_BGRNir'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Rs_predict'
# folder_output_path = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Img_origin_4band_BGRNir/Rs_predict_model_origin_model_tonghop"
# # folder_output_path_green = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia Mining Exhibition Data/Rs_predict"

# # model_path_green = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/a_model_all/gen_Green_original_Original_4band_cut_512_stride_200_time_20230515_142534_20230515_144519/gen_Green_original_Original_4band_cut_512_stride_200_time_20230515_142534_20230515_144519.h5'
# model_path_green = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG/Green/gen_ALL_Green_Original_4band_cut_512_stride_200_time_20230518_100524_20230518_100730Uint16/gen_ALL_Green_Original_4band_cut_512_stride_200_time_20230518_100524_20230518_100730Uint16.h5'
# size_model = 512
# model_green = unet_basic((size_model, size_model, 4))

# # model_path_water = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/a_model_all/gen_Water_original_Original_4band_cut_512_stride_200_time_20230515_112131_20230515_112650/gen_Water_original_Original_4band_cut_512_stride_200_time_20230515_112131_20230515_112650.h5'
# # model_path_water = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/training_with_original/a_model_all/gen_Water_original_Original_4band_cut_512_stride_200_time_20230515_112131_20230515_162207v22/gen_Water_original_Original_4band_cut_512_stride_200_time_20230515_112131_20230515_162207v22.h5'
# model_path_water = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG/Water/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_232227Uint16/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_232227Uint16.h5'
# size_model = 512
# model_water = unet_basic((size_model, size_model, 4))


# """Regis Uint16"""
# # folder_image_path_ori = r'/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Regis_4band_original/Image_origin_cog'
# # folder_output_path = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Regis_4band_original/V1/Classification05"
# model_path_water = r'/home/skm/SKM16/Planet_GreenChange/0_DataTongHopforBIG_model/DataTraining_Origin/0_ModelTongHopBIG/Water/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_232227Uint16/gen_ALL_Water_Original_4band_cut_512_stride_200_time_20230517_222750_20230517_232227Uint16.h5'
# size_model = 512
# model_water = unet_basic((size_model, size_model, 4))



# CLOUD FREE PLANET
# model_path = r'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/All_model_cloud/gen_Cloud_Uint8_4band_cut_512_stride_200_20230505_094454.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_result_mosaic/gen_Cloud_Uint8_4band_cut_512_stride_200_20230505_094454'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))



# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948.h5'
# # folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/RS_a'

# size_model = 512
# model = unet_basic((size_model, size_model, 4))