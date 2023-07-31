import rasterio
import cv2
import numpy as np
import glob, os


def read_mask(url):
    with rasterio.open(url) as src:
        array = src.read()[0]
    return array
def list_cnt_to_list_cnx(list_cnt):
    list_cnx =[]
    for i in range(len(list_cnt)):
    #    cnx = np.reshape(list_cnt[i], (1,len(list_cnt[i]),2))
        cnx = np.reshape(list_cnt[i], (len(list_cnt[i]),2))
        cnx = cnx.astype(int)
        list_cnx.append(cnx)
    return list_cnx

#im, mask, dien tich xoa => contour rm, mask _rm
def remove_area(base_path, mask_base, area):
    im2, contours, hierarchy = cv2.findContours(mask_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour2 = []
    for cnt in contours:
        if cv2.contourArea(cnt) > area:
            contour2.append(cnt)

    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
    mask_remove = np.zeros((h,w), dtype=np.uint8)
    list_cnx = list_cnt_to_list_cnx(contour2)
    cv2.fillPoly(mask_remove, list_cnx, 255)
    return contour2, mask_remove    

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id

def rm_small_area(base_path, mask_base, outputFileName, area):


    contour2, mask_remove = remove_area(base_path, mask_base, area)

    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # opening = cv2.morphologyEx(mask_remove,cv2.MORPH_GRADIENT, kernel,iterations = 10)
    # mask_remove = cv2.erode(mask_remove, kernel, iterations = 1)
    # contour2, mask_remove = remove_area(base_path, opening, area)



    with rasterio.open(base_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height

    crs = rasterio.crs.CRS({"init": "epsg:3857"})
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    # print(masking(r"/media/building/building/data_source/tmp/Malaysia-jupem/image_mask/forest.tif")[0].shape)
    new_dataset.write(mask_remove,1)
    new_dataset.close()


def main(img_path_dir, mask_path_dir, area):
    id_img_list = create_list_id(img_path_dir)
    print(id_img_list)
    parent = os.path.dirname(img_path_dir)
    foder_name = os.path.basename(img_path_dir)    
    img_procesing = os.path.join(parent,foder_name+'_proccesing800')
    
    if not os.path.exists(img_procesing):
        os.makedirs(img_procesing)

    for id_img in id_img_list:
        img_path = os.path.join(img_path_dir, id_img + '.tif')
        print(img_path)
        mask_path = os.path.join(mask_path_dir, id_img + '.tif') 
        mask_base = read_mask(mask_path) 
        outputFileName = os.path.join(img_procesing, id_img + '_rm.tif')   
        rm_small_area(img_path, mask_base, outputFileName, area)




if __name__ == "__main__":
    img_path = r"/media/khoi/Image/India/Zoom16/Image/Image1/Predict/Image1_cut"
    mask_path = r"/media/khoi/Image/India/Zoom16/Image/Image1/Predict/Image1_cut_mask"
    area = 800
    # coordinate = 3857
    
    main(img_path,mask_path,area)

