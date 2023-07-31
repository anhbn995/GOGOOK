import numpy as np
from tqdm import tqdm
import geopandas, glob, os, rasterio, cv2

path_images = "/mnt/Nam/Buildinggg/data_train_v1/train/images/*.tif"
path_shapes = "/mnt/Nam/Buildinggg/data_train_v1/shape/"
outpath_txt = "/mnt/Nam/Buildinggg/data_train_v1/labelTxt/"

list_images = glob.glob(path_images) + glob.glob("/mnt/Nam/Buildinggg/data_train_v1/val/images/*.tif")

def convert(a):
    return ((np.array(a.exterior.coords[:])[...,:2] - np.array((transform[2], transform[5])))/ np.array((transform[0], transform[4]))).astype(np.uint16)

def annToMask(ann, height, width):
    m = np.zeros((height,width), dtype=np.uint8)
    cv2.fillPoly(m, ann.astype(int), 1)
    return m


for path_image in tqdm(list_images):
    path_shape = os.path.join(path_shapes, os.path.basename(path_image).replace(".tif",'.shp'))
    with rasterio.open(path_image) as r:
        transform = r.transform
        width, height = r.width, r.height
    gdf = geopandas.read_file(path_shape)
    try:
        data = gdf['geometry'].apply(convert)
        A =[]
        for i in data:
            datax = annToMask(i[np.newaxis,...], width, height)
            cnts, hierarchy = cv2.findContours((255*datax).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            rect = cv2.minAreaRect(cnts[0])
            if rect[1][1]>rect[1][0]:
                angle = 90-rect[2]
            else:
                angle = -rect[2]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            A.extend([" ".join(str(e) for e in box.reshape(-1)) + " car" + " 1" + " \n"])
        if not A: print(gdf)
        if A:
            out_txt = os.path.join(outpath_txt, os.path.basename(path_image).replace(".tif",'.txt'))
            with open(out_txt, 'a') as file:
                file.writelines(A)
    except:
        print(path_image)
