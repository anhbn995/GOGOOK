import glob, os
import rasterio
import geopandas as gp
import multiprocessing
import rasterio.features

from functools import partial
from multiprocessing.pool import Pool



def create_mask_by_shapefile(shape_path, height, width, tr):
    list_shape = [shape_path]
    list_geometry = []
    for shape_id in list_shape: 
        print(shape_id)
        shp = gp.read_file(shape_id)
        ls_geo = [(x.geometry) for i, x in shp.iterrows()]
        list_geometry.extend(ls_geo)
    mask = rasterio.features.rasterize(list_geometry
                                    ,out_shape=(height, width)
                                    ,transform=tr)
    return mask

def arr2raster(path_out, bands, height, width, tr, dtype="uint8",crs=None,projstr=None):
    num_band = len(bands)
    new_dataset = rasterio.open(path_out, 'w', driver='GTiff',
                            height = height, width = width,
                            count = num_band, dtype = dtype,
                            crs = crs,
                            transform = tr,
                            # nodata = 0,
                            compress='lzw')
    if num_band == 1:
        new_dataset.write(bands[0], 1)
    else:
        for i in range(num_band):
            new_dataset.write(bands[i],i+1)
    new_dataset.close()

def build_mask(image_id,img_dir,path_create,path_shape):
    path_image = os.path.join(img_dir,image_id+'.tif')
    print(path_image)
    output_mask =  os.path.join(path_create,image_id+'.tif')

    with rasterio.open(path_image) as src:
        tr = src.transform
        w,h = src.width,src.height
        projstr = (src.crs.to_string())
        print(projstr)
        crs = src.crs
        check_epsg = crs.is_epsg_code
        coordinate = src.crs.to_epsg()
        img_filter = src.read_masks(1)
    path_shape_file = os.path.join(path_shape,image_id+'.shp')
    mask1 = create_mask_by_shapefile(path_shape_file, h, w, tr)*255
    mask1[img_filter==0]=0

    # TO DO : add to create cloud smaller than origin
    import cv2
    import skimage
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    mask1 = mask1.astype(bool)
    mask1 = skimage.morphology.binary_dilation(mask1, footprint=kernel) 
    # END

    arr2raster(output_mask, [mask1], h, w, tr, dtype="uint8",crs=crs,projstr=projstr)

def create_list_id(path,end_str):
    num_str = len(end_str)
    list_id = []
    os.chdir(path)
    for file in glob.glob("*{}".format(end_str)):
        list_id.append(file[:-num_str])
    return list_id
    
def main_build_mask(img_dir, shape_path):
    core = multiprocessing.cpu_count()//4
    parent = os.path.dirname(img_dir)
    foder_name = os.path.basename(img_dir)

    list_id = create_list_id(img_dir,'.tif')
    if not os.path.exists(os.path.join(parent, 'mask')):
        os.makedirs(os.path.join(parent, 'mask'))
    path_create = os.path.join(parent, 'mask')

    p_cnt = Pool(processes=core)
    p_cnt.map(partial(build_mask,img_dir=img_dir,path_create=path_create,path_shape=shape_path), list_id)
    p_cnt.close()
    p_cnt.join()
    return path_create

# if __name__ == "__main__":
    # args_parser = argparse.ArgumentParser()

    # args_parser.add_argument(
    #     '--img_dir',
    #     help='Orginal Image Directory',
    #     required=True
    # )

    # args_parser.add_argument(
    #     '--shape_dir',
    #     help='Box cut directory',
    #     required=True
    # )

    # param = args_parser.parse_args()
    # img_dir = param.img_dir
    # shape_path = param.shape_dir

    # x1 = time.time()
    # main_build_mask(img_dir,shape_path)
    # print(time.time() - x1, "second")
