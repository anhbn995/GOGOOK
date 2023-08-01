import os, glob
from tqdm import tqdm
import geopandas as gpd

# fp_shp = r"/home/skm/SKM16/Data/Uzabekittan/test.shp"
dir_shp_in = r"/home/skm/SKM16/Data/Uzabekittan/Test/out/Building"
dir_shp_out = r"/home/skm/SKM16/Data/Uzabekittan/Test/out/Building_centroid"
os.makedirs(dir_shp_out, exist_ok=True)

list_fp_shp = glob.glob(os.path.join(dir_shp_in, "*.shp"))
for fp_shp in tqdm(list_fp_shp):
    polygons = gpd.GeoDataFrame.from_file(fp_shp)
    polygons.geometry = polygons.representative_point()
    name_shp = os.path.basename(fp_shp).replace('.shp','_point_centroid_in.shp')
    polygons.to_file(os.path.join(dir_shp_out, name_shp))