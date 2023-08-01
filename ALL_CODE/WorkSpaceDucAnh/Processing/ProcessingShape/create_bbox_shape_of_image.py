import os, glob
import rasterio
from osgeo import ogr, osr
from tqdm import tqdm

def create_bbox_shapefile(image_path, output_shapefile):
    # Đọc thông tin về hình ảnh
    with rasterio.open(image_path) as src:
        transform = src.transform
        extent = src.bounds

    # Tạo đối tượng shapefile
    driver = ogr.GetDriverByName('ESRI Shapefile')
    output_ds = driver.CreateDataSource(output_shapefile)

    # Tạo hệ toạ độ từ transform của hình ảnh
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromWkt(src.crs.wkt)

    output_layer = output_ds.CreateLayer('bounding_box', spatial_reference, ogr.wkbPolygon)

    # Tạo trường để lưu trữ tên hình ảnh
    field_defn = ogr.FieldDefn('image_name', ogr.OFTString)
    output_layer.CreateField(field_defn)

    # Tạo đối tượng bounding box và thêm vào shapefile
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(extent.left, extent.bottom)
    ring.AddPoint(extent.right, extent.bottom)
    ring.AddPoint(extent.right, extent.top)
    ring.AddPoint(extent.left, extent.top)
    ring.AddPoint(extent.left, extent.bottom)

    bbox = ogr.Geometry(ogr.wkbPolygon)
    bbox.AddGeometry(ring)

    feature_defn = output_layer.GetLayerDefn()
    feature = ogr.Feature(feature_defn)
    feature.SetGeometry(bbox)
    feature.SetField('image_name', image_path)
    output_layer.CreateFeature(feature)

    # Đóng các đối tượng
    output_ds = None


def main_AOI_roi_den_image(list_fn_AOI, dir_chua_AOI, dir_chua_shp_all, run_1_img = True):
    """ ở đây đường dẫn sẽ chứa nhiều AOI
        mỗi AOI chứa nhiều ảnh khác nhau
        nhưng mặc định sẽ có chung kích thước shp
    Args:
        list_fn_AOI (_type_): _description_
        dir_chua_AOI (_type_): _description_
        dir_chua_shp_all (_type_): _description_
    """
    
    for fn_aoi in list_fn_AOI:
        dir_aoi_contain_img = os.path.join(dir_chua_AOI, fn_aoi)
        dir_aoi_contain_shp = os.path.join(dir_chua_shp_all, fn_aoi)
        os.makedirs(dir_aoi_contain_shp, exist_ok=True)
        list_fp_img = glob.glob(os.path.join(dir_aoi_contain_img, '*.tif'))
        for fp_img in tqdm(list_fp_img):
            if run_1_img:
                fp_shp_out = os.path.join(dir_aoi_contain_shp, fn_aoi + '.shp')
                create_bbox_shapefile(fp_img, fp_shp_out)
                break
            else:
                fp_shp_out = os.path.join(dir_aoi_contain_shp, os.path.basename(fp_img).replace('.tif','.shp'))
                create_bbox_shapefile(fp_img, fp_shp_out)
                
                
if __name__=="__main__":
    # Sử dụng hàm để tạo shapefile cho hình ảnh
    list_fn_AOI = ["AOI_02","AOI_07","AOI_11","AOI_14","AOI_15"]
    dir_chua_AOI = r'/home/skm/SKM16/Sentinel2_Water/DataL2A/1_Image/1_Image_raw'
    dir_chua_shp_all = r'/home/skm/SKM16/Sentinel2_Water/DataL2A/2_Shape_AOI'
    main_AOI_roi_den_image(list_fn_AOI, dir_chua_AOI, dir_chua_shp_all, run_1_img = True)

