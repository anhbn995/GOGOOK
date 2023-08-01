import os
import glob
import gdal
import rasterio
import numpy as np
import multiprocessing
from functools import partial
from multiprocessing.pool import Pool
import geopandas as gp

def reproject_image(src_path, dst_path, dst_crs='EPSG:4326'):
    with rasterio.open(src_path) as ds:
        nodata = ds.nodata or 0
    if ds.crs.to_string() != dst_crs:
        print(f'convert to {dst_crs}')
        temp_path = dst_path.replace('.tif', 'temp.tif')
        option = gdal.TranslateOptions(gdal.ParseCommandLine("-co \"TFW=YES\""))
        gdal.Translate(temp_path, src_path, options=option)
        option = gdal.WarpOptions(gdal.ParseCommandLine("-t_srs {} -dstnodata {}".format(dst_crs, nodata)))
        gdal.Warp(dst_path, temp_path, options=option)
        os.remove(temp_path)
    else:
        import shutil
        print(f'coppy image to {dst_path}')
        shutil.copyfile(src_path, dst_path)
        print('done coppy')
    return True


def create_list_id(path):
    list_image = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".tif"):
                list_image.append(os.path.join(root, file))
    return list_image


def cal_bancut(image_path,num_channel):
    dataset = gdal.Open(image_path)
    band_cut_th = {k: dict(max=0, min=0) for k in range(num_channel)}
    for i_chan in range(num_channel):
        values_ = dataset.GetRasterBand(i_chan+1).ReadAsArray().astype(np.float16)
        values_[values_==0] = np.nan
        band_cut_th[i_chan]['max'] = np.nanpercentile(values_, 98)
        band_cut_th[i_chan]['min'] = np.nanpercentile(values_, 2)
    # print(band_cut_th[i_chan]['max'])
    # print(band_cut_th[i_chan]['min'])
    return band_cut_th


def buil_3_band(image_path, path_create, num_channel):
    list_value = [[1.0, 2177.0],
                 [1.0, 2274.0],
                 [1.0, 2197.0],
                 [1.0, 3570.0]]
    dir_name = os.path.basename(os.path.dirname(image_path))
    image_name = os.path.basename(image_path)[:-4]
    path_out = os.path.join(path_create,dir_name)
    output = os.path.join(path_out,image_name+'.tif')
    band_cut_th = cal_bancut(image_path,num_channel)
    options_list = ['-ot UInt16','-a_nodata 0','-colorinterp_4 undefined']
    for i_chain in range(num_channel):
        options_list.append('-b {}'.format(i_chain+1))
    for i_chain, value in zip(range(num_channel),list_value):
        options_list.append('-scale_{} {} {} {} {} -exponent_{} 1.0'.format(i_chain+1,band_cut_th[i_chain]['min'],band_cut_th[i_chain]['max'],value[0],value[1],i_chain+1))
    options_string = " ".join(options_list)
    gdal.Translate(output,
            image_path,
            options=options_string)


def norm_data(folder_path, num_channel):
    parent = os.path.dirname(folder_path)
    # foder_name = os.path.basename(foder_path)
    core = multiprocessing.cpu_count()//4
    list_id = create_list_id(folder_path)

    path_create = os.path.join(parent.replace(os.path.basename(parent), 'norm_data'))
    if not os.path.exists(path_create):
        os.makedirs(path_create)

    for image_path1 in list_id:
        dir_name = os.path.basename(os.path.dirname(image_path1))
        path_out = os.path.join(path_create,dir_name)
        if not os.path.exists(path_out):
            os.makedirs(path_out)
            
    num_norm_data = len(glob.glob(os.path.join(path_out, '*.tif')))
    num_crop_data = len(list_id)
    if num_norm_data!=num_crop_data:
        p_cnt = Pool(processes=core)
        p_cnt.map(partial(buil_3_band,path_create=path_create,num_channel=num_channel), list_id)
        p_cnt.close()
        p_cnt.join()
    return path_out


def cut(img_name, img_dir, box_dir, img_cut_dir):
    print("ok")
    image_path = os.path.join(img_dir,img_name+'.tif')
    shape_path = glob.glob(os.path.join(box_dir,'*.shp'))

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
    # img_path = r''
    # box_path = r''
    # tmp_path = r''
    # main_cut_img(img_path, box_path, tmp_path)
    
    dir_box = r"/home/skm/SKM16/Planet_GreenChange/Data_original/box_green"
    # dir_img = r"/home/skm/SKM16/A_CAOHOC/img_unit8"
    dir_img = r'/home/skm/SKM16/Planet_GreenChange/Data_4band_Green/ImgRGBNir_8bit_perimage'
    list_fp_have_box = glob.glob(os.path.join(dir_box, "*.shp"))
    # tmp_path = os.path.join(dir_img,"cut_box_Grass")
    # os.makedirs(tmp_path)
    for fp_box in list_fp_have_box:
        fn = os.path.basename(fp_box).replace(".shp","")
        fp_img = os.path.join(dir_img, fn +  ".tif")
        print(fp_img)
        print(fp_box)
        main_cut_img(fp_img, fp_box, dir_img)
    
    