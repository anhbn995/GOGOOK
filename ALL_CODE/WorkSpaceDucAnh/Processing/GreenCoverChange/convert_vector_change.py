import os, glob
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from tqdm import tqdm
from geopandas import GeoDataFrame

# out_dir = os.path.join(dir_out, 'shp')
dir_out = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_result_mosaic/RS_final/clip_by_aoi/Rs_change'
out_dir_shape = r'/home/skm/SKM16/Planet_GreenChange/1_Real_dataSet/All_image_origin/all_result_mosaic/RS_final/clip_by_aoi/Rs_change_shp'

os.makedirs(out_dir_shape, exist_ok=True)

for fp in tqdm(glob.glob(os.path.join(dir_out,'*.tif'))):
    fname_shp = os.path.basename(fp).replace('.tif','.shp')
    
    with rasterio.open(fp) as src:
        data = src.read(1, masked=True)
        # print(data)
        # Use a generator instead of a list
        shape_gen = ((shape(s), v) for s, v in shapes(data, transform=src.transform))
        # print(shape_gen)
        # or build a dict from unpacked shapes
        gdf = GeoDataFrame(dict(zip(["geometry", "class"], zip(*shape_gen))), crs=src.crs)
    gdf['class'] = gdf['class'].replace([1], 'Green to Non')
    gdf['class'] = gdf['class'].replace([2], 'Non to Green')
    
    gdf.to_file(os.path.join(out_dir_shape, fname_shp))