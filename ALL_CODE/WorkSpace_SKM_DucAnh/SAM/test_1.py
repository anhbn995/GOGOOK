import os
import torch
from samgeo import SamGeo
import geopandas as gpd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_sam_point(model_path, img_path, input_prompt_shp, outputVector, dir_tmp):
    # check shp point
    df_prompt = gpd.read_file(input_prompt_shp)
    if df_prompt.geom_type.unique()[0] == 'Point' and 'id' in df_prompt.columns:
        os.makedirs(os.path.dirname(outputVector), exist_ok=True)
        os.makedirs(dir_tmp, exist_ok=True)
        
        sam = SamGeo(
                checkpoint=model_path,
                model_type="vit_h", # vit_l, vit_b , vit_h
                automatic=False,
                device=device,
                sam_kwargs=None,
            )
        sam.set_image(img_path)
        df_point = df_prompt['geometry']
        # Convert each geometry element to a pair of lists
        point_lists = []
        for geometry in df_point:
            point_lists.append([geometry.coords[0]])
        point_labels = df_prompt['id'].to_list()
        
        print(len(point_labels))
        print(len(point_labels), point_labels)
        point_labels = point_labels[1]
        point_lists = point_lists[1]
        
        epsg_code = "EPSG:" + str(df_prompt.crs.to_epsg())
        outputRaster_tmp = os.path.join(dir_tmp, os.path.basename(img_path))
        sam.predict(point_lists, point_labels=point_labels, point_crs=epsg_code, output=outputRaster_tmp)
        sam.raster_to_vector(outputRaster_tmp, outputVector)#, simplify_tolerance=0.00001)
    else:
        print('shapefile dont have "id" field or Point geometry')
  
model_path = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/sam_vit_h_4b8939.pth"
img_path = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/a_4326.tif'
input_prompt_shp = r'/home/skm/public_mount/tmp_Duy/point_sam_2/point_sam_2.shp'
outputVector = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/TEST_DOCKER/meo1.shp'
dir_tmp = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/TEST_DOCKER/tmp1'

run_sam_point(model_path, img_path, input_prompt_shp, outputVector, dir_tmp)