import rasterio
import numpy as np
import regex


def get_index_nodata(in_fp_img):
    """ Ham nay lay ra vi tri cua nodata

    Args:
        in_fp_img (str): duong dan anh goc

    Returns:
        _type_: index cua cac vung nodata
    """
    with rasterio.open(in_fp_img) as src:
        red = src.read(1)
        green = src.read(2)
        blue = src.read(3)
        nir = src.read(4)
        nodata_value = src.nodata

        red_mask = red == nodata_value
        green_mask = green == nodata_value
        blue_mask = blue == nodata_value
        nir_mask = nir == nodata_value

        # Kết hợp các mask lại với nhau
        combined_mask = red_mask & green_mask & blue_mask & nir_mask
        return np.where(np.array([combined_mask])==True)


def find_index_class_flow_name_class(dict_fp_class_and_value, name_class):
    """_summary_

    Args:
        dict_fp_class_and_value (_type_): _description_
        name_class (_type_): _description_
    """
    all_key =  list(dict_fp_class_and_value.keys())
    result = next((s for s in all_key if s.startswith(name_class)), None)
    
    if result:
        fp_img = dict_fp_class_and_value[result]
        with rasterio.open(fp_img) as src:
            img =  src.read()
        index_class = np.where(img != 0)
    else:
        index_class = None
    return index_class
    

def union_class_water_green_buildup_nodata(in_fp_origin_img, dict_fp_class_and_value, list_thu_tu_uu_tien):
    
    """ Gop cac lop lai vs nhau de co the thanh mot class hoan chinh

    Args:
        in_fp_origin_img (_type_): _description_
        dict_fp_class_and_value = {
                                    "green1": None,
                                    "water2": None,
                                    "buildUp3": None,
                                    "forest4": None,
                                    "cloud5": None
                                    }
        list_thu_tu_uu_tien: la danh sanh theo tu tu la do uu tien giam giam [cloud, water, green, forest, buildup], \
            y/c ten class phai giong vs vs key cua class
        
    """
    ind_nodata = get_index_nodata(in_fp_origin_img)
    
    for class_first in list_thu_tu_uu_tien:
        with rasterio.open(fp_class) as src:
            img = src.read()
            meta = src.meta
        
    
    