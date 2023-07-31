import glob, os
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
import shutil
import rasterio
import rasterio.mask
import geopandas as gp

def cut(img_name, img_dir, box_dir, img_cut_dir):
    image_path = os.path.join(img_dir,img_name+'.tif')
    shape_path = glob.glob(os.path.join(box_dir,'*.shp'))[0]

    with rasterio.open(image_path, mode='r+') as src:
        projstr = src.crs.to_string()
        # print(projstr)
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
            # print(epsg_code)
        else:
            epsg_code = None
    if epsg_code:
        out_crs = {'init':'epsg:{}'.format(epsg_code)}
    else:
        out_crs = projstr
    bound_shp = gp.read_file(shape_path)
    bound_shp = bound_shp.to_crs(out_crs)

    for index2, row_bound in bound_shp.iterrows():
        geoms = row_bound.geometry
        img_cut = img_name+"_{}.tif".format(index2)
        img_cut_path = os.path.join(img_cut_dir, img_cut)
        # if os.path.exists(img_cut_path):
        #     pass
        # else:
        try:
            if not os.path.exists(img_cut_path):
                with rasterio.open(image_path,BIGTIFF='YES') as src:
                    out_image, out_transform = rasterio.mask.mask(src, [geoms], crop=True)
                    out_meta = src.meta
                # print( "height",out_image.shape[1])
                out_meta.update({"driver": "GTiff",
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform})
                with rasterio.open(img_cut_path, "w", **out_meta) as dest:
                    dest.write(out_image)
        except :
            pass
                # raise Exception("Error crop image")

def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
    return list_id

def main_cut_img(img_path, box_path, tmp_path): 
    core = multiprocessing.cpu_count()//4
    img_list = create_list_id(img_path)

    img_cut_dir = tmp_path+'_cut'

    print("Run crop image with aoi ...")  
    if not os.path.exists(img_cut_dir):
        os.makedirs(img_cut_dir)    

    p_cnt = Pool(processes=core)    
    p_cnt.map(partial(cut,img_dir=img_path,box_dir=box_path,img_cut_dir=img_cut_dir), img_list)
    p_cnt.close()
    p_cnt.join()    
    print("Done")
    return img_cut_dir

if __name__=='__main__':
    img_path = '/home/quyet/DATA_ML/Projects/GreenCover_Npark/8bit_perimage/base'
    box_path = '/home/quyet/DATA_ML/Projects/GreenCover_Npark/8bit_perimage/box'
    tmp_path = '/home/quyet/DATA_ML/Projects/GreenCover_Npark/8bit_perimage/tmp'

    main_cut_img(img_path,box_path,tmp_path)