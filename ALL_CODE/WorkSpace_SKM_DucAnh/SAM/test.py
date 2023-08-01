# import torch
# from samgeo import SamGeo
# import os,sys

# chkpnt_dir = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/"
# # sam_vit_h_4b8939
# # sam_vit_l_0b3195
# # sam_vit_b_01ec64
# checkpoint = os.path.join(chkpnt_dir, "sam_vit_h_4b8939"+".pth")
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# #Run SamGeo model
# sam = SamGeo(
#     checkpoint=checkpoint,
#     model_type="vit_h", # vit_l, vit_b , vit_h
#     automatic=True,
#     device=device,
#     sam_kwargs=None,
# )

# img_path = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/POC_AOI_Here_VN_Final_Image_10CM.tif"
# outputRaster = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/A_Test_xin/POC_AOI_Here_VN_Final_Image_10CM_3x3.tif"
# outputGPKG = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/A_Test_xin/POC_AOI_Here_VN_Final_Image_10CM_3x3.gpkg"
# outputVector = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/A_Test_xin/POC_AOI_Here_VN_Final_Image_10CM_3x3.shp"

# # with rasterio.open(img_path) as data:
# #     inRaster = data.read().transpose(2,1,0)
# # Run SamGeoPredictor#Generate Mask
# sam.generate(source=img_path, output=outputRaster,batch=True, erosion_kernel=(3,3))
# sam.tiff_to_gpkg(outputRaster, outputGPKG, simplify_tolerance=0.00001)

# outputVector = r"/home/skm/public_mount/tmp_Duy/aa.shp"
# sam.raster_to_vector(outputRaster, outputVector, simplify_tolerance=0.00001)


# """a"""
import torch
from samgeo import SamGeo
import os,sys
chkpnt_dir = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/"
checkpoint = os.path.join(chkpnt_dir, "sam_vit_h_4b8939"+".pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_path = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/a_4326.tif"
point_shape = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/a_4326.shp'
outputRaster = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/a_ok_1.tif"
import geopandas as gpd
gdf = gpd.read_file(point_shape)

point = gdf['geometry'].iloc[0]
coords = point.coords[0]
point_lists = [list(coords)]
print(point_lists)

# with rasterio.open(img_path) as data:
#     inRaster = data.read().transpose(2,1,0)

"""Training point"""
sam = SamGeo(
    checkpoint=checkpoint,
    model_type="vit_h", # vit_l, vit_b , vit_h
    automatic=False,
    device=device,
    sam_kwargs=None,
)
sam.set_image(img_path)
print(point_shape)
sam.predict(point_shape, point_labels=1, point_crs="EPSG:4326", output=outputRaster)


# """Training Langue"""
import torch
import os
from samgeo.text_sam import LangSAM
# sam_vit_h_4b8939
# sam_vit_l_0b3195
# sam_vit_b_01ec64

chkpnt_dir = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/"
checkpoint = os.path.join(chkpnt_dir, "sam_vit_h_4b8939"+".pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = LangSAM()

text_prompt = "rooftop"
image = r"/home/skm/public_mount/tmp_Duy/POC_AOI_Here_VN_Final_Image_15CM.tif"
import os
dir_out = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/POC_AOI_Here_VN_Final_Image_10CM_aaaaa"
os.makedirs(dir_out, exist_ok=True)
sam.predict(image, text_prompt, box_threshold=0.24, text_threshold=0.24, return_results=True)

sam.show_anns(
    cmap='Greys_r',
    add_boxes=False,
    alpha=1,
    title='Automatic Segmentation of Trees',
    blend=False,
    output=os.path.join(dir_out, "tree.tif"),
)
sam.raster_to_vector(os.path.join(dir_out, "tree.tif"), os.path.join(dir_out, "tree.shp"))