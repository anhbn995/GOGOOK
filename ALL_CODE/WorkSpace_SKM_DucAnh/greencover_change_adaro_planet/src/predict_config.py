
""" Du lieu Planet """

image_path = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/2023-05_mosaic.tif'
folder_rs = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/RS'

model_path_object = {
    "water_0": r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230427_104050.h5',
    "green_1": r'/home/skm/SKM16/Planet_GreenChange/Tong_hop_model/gen_Green_UINT8_4band_cut_512_stride_200_20230428_152806_V2_green.h5'
}



""" Main chinh """
import re, os
def get_name_object_and_value(string):
    pattern = r"([a-zA-Z, 0-9]+)_([0-9]+)"
    try:
        match = re.match(pattern, string)
        name_class = match.group(1)
        value_class = match.group(2)
        return name_class, value_class
    except: 
        raise "Dat ten cua doi tuong model sai cai truc ex: A_1"
        
dict_fp_rs_class = dict()
for key_name in model_path_object:
    name_class, value_class = get_name_object_and_value(key_name)
    fp_out_class = os.path.join(folder_rs, os.path.basename(image_path).replace('.tif', '_' + name_class + '.tif'))
    dict_fp_rs_class[name_class] = fp_out_class

    