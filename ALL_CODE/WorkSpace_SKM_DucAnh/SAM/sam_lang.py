import os
import torch
from samgeo.text_sam import LangSAM

chkpnt_dir = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/"
checkpoint = os.path.join(chkpnt_dir, "sam_vit_h_4b8939"+".pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = LangSAM()

text_prompt = "tree"
image =  r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/img7x1/Bi_sai_tree.tif"
dir_out = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/img7x1/SAM_LANGE_FIX_OOM"
os.makedirs(dir_out, exist_ok=True)
sam.predict(image, text_prompt, box_threshold=0.27, text_threshold=0.33, output=os.path.join(dir_out, f"{text_prompt}_vnn.tif"), return_results=True)
name_file = os.path.basename(image).replace('.tif', '_' + text_prompt + '.tif')


# sam.show_anns(
#     cmap='Greys_r',
#     add_boxes=False,
#     alpha=1,
#     title='Automatic Segmentation of Trees',
#     blend=False,
#     output=os.path.join(dir_out, f"{text_prompt}.tif"),
# )
# sam.raster_to_vector(os.path.join(dir_out, f"{text_prompt}.tif"), os.path.join(dir_out, f"{text_prompt}.shp"))