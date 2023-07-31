import rasterio
import cv2
import numpy as np
import os, glob


def get_list_name_file(path_folder, get_full_path = True, name_file = '*.tif'):
    list_img_dir = []
    for file_ in glob.glob(os.path.join(path_folder, name_file)):
        if get_full_path:
            list_img_dir.append(file_)
        else:
            _, tail = os.path.split(file_)
            list_img_dir.append(tail)
    return list_img_dir

def convert_mask_012(path_image, path_out):
    src = rasterio.open(path_image)
    img = src.read()
    img[np.where(img==255)] = 2
    img = img[0]
    _, contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for i in contours[0]:
        a,b = i[0]
        img[(b,a)]=1
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))   
    img = cv2.dilate(img,kernel,iterations = 1)
    with rasterio.open(
        path_out,
        'w',
        driver='GTiff',
        height=src.height,
        width=src.width,
        count=src.count,
        dtype=img.dtype,
        crs=src.crs,
        transform=src.transform,
    ) as dst:
        dst.write(np.array([img]))
        
def main(path_folder):
    list_name = get_list_name_file(path_folder,False)
    out_folder = path_folder+"_mask_012"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    for name in list_name:
        path_image = os.path.join(path_folder, name)
        path_out = os.path.join(out_folder, name)
        convert_mask_012(path_image, path_out)
    
if __name__ == '__main__':
    path_folder = r"/mnt/66A8E45DA8E42CED/farm_singlefarm/img_cut_box_mask"
    main(path_folder)
    