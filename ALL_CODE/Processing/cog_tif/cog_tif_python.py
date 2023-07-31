import os, glob
import subprocess


def get_list_fp(folder_dir, type_file = '*.TIF'):
    """
        Get all file path with file type is type_file.
    """
    list_fp = []
    for file_ in glob.glob(os.path.join(folder_dir, type_file)):
        head, tail = os.path.split(file_)
        list_fp.append(os.path.join(head, tail))
    return list_fp


path_dir = r"/home/skm/SKM/WORK/KhongGianXanh/1_IMG_ORGIN/Create_data_train/img_orgin"
list_fp = get_list_fp(path_dir)
for fp in list_fp:
    tmp_dir = os.path.join(path_dir, "tmp")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    result_dir = os.path.join(path_dir, "img_orgin_COG")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    fp_tmp = os.path.join(tmp_dir, os.path.basename(fp))
    fp_rs = os.path.join(result_dir, os.path.basename(fp))

    run1 = f"gdal_translate {fp} {fp_tmp} -co TILED=YES -co COMPRESS=DEFLATE"
    run2 = f"gdaladdo -r nearest {fp_tmp} 2 4 8 16 32"
    run3 = f"gdal_translate {fp_tmp} {fp_rs} -co TILED=YES -co COMPRESS=DEFLATE -co COPY_SRC_OVERVIEWS=YES"
    # print(run1.split(" "))
    subprocess.run(run1.split(" "))
    subprocess.run(run2.split(" "))
    subprocess.run(run3.split(" "))
