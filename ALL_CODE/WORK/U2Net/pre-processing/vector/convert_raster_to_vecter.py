import rasterio
import numpy as np
from math import sqrt
import timeit, time, os, glob
import Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

path_img = '/mnt/data/public/farm-bing18/Bingmaps_wajo_predict/05_01/model_u2net/mask/*.tif'
# for img_path in os.listdir(path_img):
for img_path in glob.glob(path_img):
    # out_img = '/mnt/data/public/farm-bing18/BingMaps_Kediri_predict/30_12/model_farm/skeleton/'+img_path
    # if not os.path.exists(img_path):
    with rasterio.open(img_path) as f:
        data = f.read()
        out_meta = f.meta
        transform = f.transform
        projstr = f.crs.to_string()
    star = time.time()
    data = remove_small_holes(data.astype(bool), area_threshold=77)
    data = remove_small_objects(data, min_size=77)
    skeleton = skeletonize(data.astype(np.uint8))
    # print(out_img)
    # with rasterio.open(out_img, "w", **out_meta, compress='lzw') as dest:
    #     dest.write(skeleton)
        
    # path_img = "/mnt/data/Nam_work_space/30_12/aaa/"
    # for img_path in os.listdir(path_img):
    #     image = path_img+img_path
    #     with rasterio.open(image) as inds:
    # img = inds.read()[0]
    # transform = inds.transform
    # projstr = inds.crs.to_string()
    # save_path = "/mnt/data/Nam_work_space/30_12/aaa/" + img_path.split('.')[0]+'.geojson'
    save_path = img_path.replace('.tif','.geojson')
    # if not os.path.exists(save_path):
    try:
        print(save_path)
        test = Vectorization.save_polygon(np.pad(skeleton[0], pad_width=1).astype(np.intc), 3,5,transform, projstr, save_path)
        print(time.time()-star)
    except:
        print(100*'-')
        print(save_path)
        print(100*'-')