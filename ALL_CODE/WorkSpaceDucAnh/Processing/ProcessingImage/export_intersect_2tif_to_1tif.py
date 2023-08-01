import rasterio
# Chuyển đổi vùng chồng lấp sang tọa độ của tệp ảnh thứ hai
from rasterio.warp import transform_bounds
import numpy as np


def get_bbox_intersect_2_img(list_fp_img):
    """ Tim giao nhau giua 2 anh va tra ve bbox cua vung giao nhau

    Args:
        list_fp_img (list): gom 2 cai anh tif, nen la cung toa do nhe

    Returns:
        _type_: tuple, bbox cua giao nhau
    """
    list_bound = []
    list_crs = []
    for fp_img in list_fp_img:
        with rasterio.open(fp_img) as src:
            bounds = src.bounds
            crs = src.crs
        list_bound.append(bounds)
        list_crs.append(crs)
        
    # Tìm vùng chồng lấp của hai tệp ảnh        
    bound_left = [bounds.left for bounds in list_bound]
    bound_bottom = [bounds.bottom for bounds in list_bound]
    bound_right = [bounds.right for bounds in list_bound]
    bound_top = [bounds.top for bounds in list_bound]

    xmin = max(bound_left)
    ymin = max(bound_bottom)
    xmax = min(bound_right) 
    ymax = min(bound_top)
    
    # Chuyển đổi vùng chồng lấp sang tọa độ của tệp ảnh thứ hai
    leftw, bottomw, rightw, topw = transform_bounds(list_crs[0], list_crs[1], xmin, ymin, xmax, ymax)
    left, bottom, right, top = transform_bounds(list_crs[1], list_crs[0], leftw, bottomw, rightw, topw)
    
    # In ra vùng chồng lấp của hai tệp ảnh
    print("Vùng chồng lấp của hai tệp ảnh là: ", left, bottom, right, top)
    return (left, bottom, right, top)


def clip_raster_by_bbox(input_path, bbox, output_path= None, return_ = True, export_file_tiff = True):
    """ Cat anh theo bbox

    Args:
        input_path (string): duong dan anh muon cat
        output_path (string): duong dan anh duoc cat
        bbox (tuple): bbox o dang tuple
        return_ (bool, optional):  co tra ve band anh va meta khong. Defaults to True.
        export_file_tiff (bool, optional): co muon xuat anh ra khong. Defaults to True.

    Returns:
        _type_: _description_
    """

    with rasterio.open(input_path) as src:
        # Get the window to read from
        minx, miny, maxx, maxy = bbox
        window = src.window(minx, miny, maxx, maxy)

        # Calculate the width and height of the output
        width = window.width
        height = window.height

        # Compute the transform for the output
        transform = rasterio.windows.transform(window, src.transform)

        # Update the metadata for the output
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': transform
        })
        if export_file_tiff:
        # Read and write the data
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(src.read(window=window))
        if return_:
            return src.read(window=window), meta
 
 
def get_dtype_big(list_dtype):
    print(list_dtype)
    max_dtype = list_dtype[0]
    for dtype_ in list_dtype[1:]:
        if np.dtype(max_dtype).itemsize < np.dtype(dtype_).itemsize:
            max_dtype = dtype_
    return max_dtype
     

def get_dtype_big_from_meta(list_meta):
    list_dtype = []
    for meta in list_meta:
        list_dtype.append(meta['dtype'])
    return get_dtype_big(list_dtype)    


def stack_image_same_size(out_fp, list_source_data, list_meta_=None):
    """

    Args:
        list_source_data (_type_): _description_
        meta (_type_, optional): _description_. Defaults to None.
    """
    
    # check list
    if type(list_source_data[0]) is str:
        type_data = "duong dan"
    else:
        type_data = False
        print("Phai co meta do!!!!!!!!")
    
    list_raster = []
    list_meta = []
    if type_data:
        for fp_img in list_source_data:
            with rasterio.open(fp_img) as src:
                list_meta.append(src.meta)
                list_raster.append(src.read())
    else:
        list_raster = list_source_data
        list_meta = list_meta_
        
    print(list_meta, "zz")
    dtype_need = get_dtype_big_from_meta(list_meta)
    img_new = np.concatenate(list_raster, axis=0)
    c,_,_ = img_new.shape
    
    meta = list_meta[0]
    meta.update({
        'count': c,
        'dtype': dtype_need
        })
    with rasterio.open(out_fp, 'w', **meta) as dst:
        dst.write(img_new)
    
        
if __name__=="__main__":
    # for name in ['20180310.tif', '20180409.tif', '20181031.tif']:
    #     img_sen = f'/home/skm/SKM16/X/Test/img_origin/{name}'
    #     img_slop = r'/home/skm/SKM16/X/Test/Landslide_Sentinel-2_DAnh/DaBac/Slope_10m.tif'
    #     out_fp = f'/home/skm/SKM16/X/Landslide_Sentinel-2_DAnh/Training_ver_5band_goc/All_img_stack_big/img_big_test_stack/stack_{name}'
        
    #     list_fp_img = [img_sen, img_slop]
        
    #     bbox = get_bbox_intersect_2_img(list_fp_img)
        
    #     list_raster = []
    #     list_meta = []
        
    #     for fp in list_fp_img:
    #         img, meta = clip_raster_by_bbox(fp, bbox, return_ = True, export_file_tiff = False)
    #         img[img<0] = 0
    #         list_raster.append(img)
    #         list_meta.append(meta)
    #     # print(list_raster)
    #     # print(list_meta)
    #     stack_image_same_size(out_fp, list_raster, list_meta_=list_meta)
    
    
    """Kieu 2"""

    # img_sen = r'/home/skm/SKM16/Tmp/XONG_XOAAAAAAAAAAAAAAAAAAAAAAAA/Img/S1A_IW_GRDH_1SDV_20220323T215024_0.tif'
    # img_slop = r'/home/skm/SKM16/Tmp/XONG_XOAAAAAAAAAAAAAAAAAAAAAAAA/Img/S1A_IW_GRDH_1SDV_20220416T215025_0.tif'
    # out_fp = f'/home/skm/SKM16/Tmp/XONG_XOAAAAAAAAAAAAAAAAAAAAAAAA/Img/align/stack.tif'
    
    
    img_sen = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/mosaic_rs/clip052023.tif'
    img_slop = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_result_mosaic/RS_final/clip_by_aoi/2023-04_mosaic_RS.tif'
    out_fp = f'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/2023-05_8bit_perimage/2023-05/mosaic_rs/stack.tif'
    
    list_fp_img = [img_sen, img_slop]
    
    bbox = get_bbox_intersect_2_img(list_fp_img)
    
    list_raster = []
    list_meta = []
    
    for fp in list_fp_img:
        img, meta = clip_raster_by_bbox(fp, bbox, return_ = True, export_file_tiff = False)
        img[img<0] = 0
        list_raster.append(img)
        list_meta.append(meta)
    # print(list_raster)
    # print(list_meta)
    stack_image_same_size(out_fp, list_raster, list_meta_=list_meta)