from datetime import datetime
now = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')


# crop_size = 256
# stride_size = 64
# percent = 0.15
# dir_img = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/2Ver3_nghiemchinh/Data_Train_and_Model/images_per95_cut_img"
# dir_mask = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/2Ver3_nghiemchinh/Data_Train_and_Model/images_per95_cut_img_mask"
# dir_out = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/2Ver3_nghiemchinh/Data_Train_and_Model/crop256_stride64_giamanhden"
# dir_out_img, dir_out_mask = crop_img_and_mask(dir_img, dir_mask, dir_out, crop_size, stride_size, percent)

"""Openland"""
# crop_size = 512
# stride_size = 256
# percent = 0.15
# dir_img = r"/home/skm/SKM16/Work/OpenLand/1_Data_train/img_train__khoang_255/img"
# dir_mask = r"/home/skm/SKM16/Work/OpenLand/1_Data_train/img_train__khoang_255/mask"
# dir_out = f"/home/skm/SKM16/Work/OpenLand/1_Data_train/img_train__khoang_255/DS_Train/crop{crop_size}_stride{stride_size}_{now}"

# BF va Road
crop_size = 512
stride_size = 256
percent = 0.15
dir_img = r"/home/skm/SKM16/Work/OpenLand/1_Data_train/img_train__khoang_255/img"
dir_mask = r"/home/skm/SKM16/Work/OpenLand/1_Data_train/img_train__khoang_255/mask_nha_va_duong/img_mask_bf_road_zadd"
dir_out = f"/home/skm/SKM16/Work/OpenLand/1_Data_train/img_train__khoang_255/DS_Train_Road_and_BF/crop{crop_size}_stride{stride_size}_{now}"



"""OPEN LAND DICH HISTOGRAM"""
