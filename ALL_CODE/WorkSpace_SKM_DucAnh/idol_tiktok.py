import os
import rasterio
import geopandas as gpd


def standardized_shape_epsg(fp_shp, image_path, out_dir):
    output_path = os.path.join(out_dir, os.path.basename(fp_shp).replace('.shp', '_std.shp'))
    os.makedirs(out_dir, exist_ok=True)
    df_shape = gpd.read_file(fp_shp)
    
    with rasterio.open(image_path, mode='r+') as src:
        projstr = src.crs.to_string()
        check_epsg = src.crs.is_epsg_code
        if check_epsg:
            epsg_code = src.crs.to_epsg()
        else:
            epsg_code = None
        if epsg_code:
            out_crs = {'init':'epsg:{}'.format(epsg_code)}
        else:
            out_crs = projstr
    gdf = df_shape.to_crs(out_crs)
    gdf.to_file(output_path)
        
    
image_dir = r'E:\TMP_XOA\mongkos\mongkos part 3_transparent_mosaic_group1.tif'
label_dir = r'E:\TMP_XOA\mongkos_2\label.shp'
box_dir = r'E:\TMP_XOA\mongkos_2\box.shp'
out_dir = r'E:\TMP_XOA\mongkos_std'
standardized_shape_epsg(label_dir, image_dir, out_dir)