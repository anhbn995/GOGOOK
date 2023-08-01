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

# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))


# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948.h5'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Water_UINT8_4band_cut_512_stride_200_20230428_091948'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))


# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green.h5'
# # folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8/all_result_mosaic/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))


# gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water.h5
# model_path = r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water.h5'
# # folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage'
# # folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_img_mosaic/img_ori_8bit_perimage/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water'
# folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8'
# folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_result_mosaic/gen_Water_UINT8_4band_cut_512_stride_200_20230429_101659_V2_water'
# size_model = 512
# model = unet_basic((size_model, size_model, 4))



# CLOUD FREE PLANET
model_path = r'/home/skm/SKM16/Data/Planet/Cloud_planet/cloud_iou/All_model_cloud/gen_Cloud_Uint8_4band_cut_512_stride_200_20230505_094454.h5'
folder_image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_mosaic_uint8'
folder_output_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_result_mosaic/gen_Cloud_Uint8_4band_cut_512_stride_200_20230505_094454'
size_model = 512
model = unet_basic((size_model, size_model, 4))