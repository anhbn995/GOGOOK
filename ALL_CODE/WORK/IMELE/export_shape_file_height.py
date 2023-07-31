import fiona
from rasterstats import zonal_stats


# in_shp_building='./test/RGB_54366543.shp'
# out_shape_height_building='./test/RGB_54366543_height.shp'
# in_img_height = r'/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE/test2/RGB_54366543_height_tm.tif'
 
def export_shp_height(in_img_height, in_shp_building, out_shape_height_building):
    with fiona.open(in_shp_building) as src:
        zs = zonal_stats(src, in_img_height, stats='median', nodata=-32768)
        size=len(zs)
        schema=src.schema
        schema['properties']['height'] = 'float'
        
        with fiona.open(out_shape_height_building, 'w', crs=src.crs, driver=src.driver, schema=schema) as dst:
            for idx, f in enumerate(src):
                print(idx)
                f['properties'].update(height=zs[idx]['median'])
                dst.write(f)
                print(f'Feature {idx}/{size}')
                
if __name__=='__main__':
    in_img_height = r'/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE/test_new/okaa.tif'
    in_shp_building = r'/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE/test_new/building_rs.shp'
    out_shape_height_building = r'/home/skm/SKM/WORK/ALL_CODE/WORK/IMELE/test_new/bbbbb.shp'
    export_shp_height(in_img_height, in_shp_building, out_shape_height_building)