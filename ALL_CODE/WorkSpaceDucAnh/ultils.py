import shutil
import geopandas as gpd

def read_txt(fp_txt):
    """
        Lấy thông tin từ txt ra và mỗi dòng là 1 phần tử của list
    """
    with open(fp_txt) as Lines:
        list_file_choosen = [x.rstrip() for x in Lines]
    return list_file_choosen


def get_info_txt_with_line_to_list(fp_txt):
    """
        Lấy thông tin từ txt ra và mỗi dòng là 1 phần tử của list, gán thêm '.tif' vào đuôi mỗi dòng
        
        Input: Đường dẫn file txt
        Output: list các dòng trong file txt và thêm '.tif'
    """
    with open(fp_txt) as Lines:
        list_file_choosen = [x.rstrip() + '.tif' for x in Lines]
    return list_file_choosen


def coppy_file_to_dir(fp_coppy, dir_dest):
    shutil.copy2(fp_coppy, dir_dest)


def remove_polygon_by_area(fp_shp, fp_out_shp, area):
    """
        Xóa những polygon trong "fp_shp" có diện tích nhỏ hơn "area"

        Input: 
        Output: file shape với đường dẫn "fp_out_shp"
    """
    df_in = gpd.read_file(fp_shp)
    df_out = df_in[df_in.area > area]
    df_out.to_file(fp_out_shp)
