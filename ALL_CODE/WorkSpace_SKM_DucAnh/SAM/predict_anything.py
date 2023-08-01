import os
import rasterio
import rasterio.windows
import numpy as np

import torch
from samgeo import SamGeo
import os,sys

chkpnt_dir = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/"
# sam_vit_h_4b8939
# sam_vit_l_0b3195
# sam_vit_b_01ec64
checkpoint = os.path.join(chkpnt_dir, "sam_vit_h_4b8939"+".pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Run SamGeo model
sam = SamGeo(
    checkpoint=checkpoint,
    model_type="vit_h", # vit_l, vit_b , vit_h
    automatic=True,
    device=device,
    sam_kwargs=None,
)


def read_image_in_windows(filename, window_size, out_fp):
    with rasterio.open(filename) as dataset:
        width = dataset.width
        height = dataset.height
        with rasterio.open(out_fp, 'w', driver='GTiff', width=width, height=height, count=1, dtype='uint8') as dst:
            num_windows_x = width // window_size + (1 if width % window_size != 0 else 0)
            num_windows_y = height // window_size + (1 if height % window_size != 0 else 0)
            print(num_windows_x, num_windows_y)
            for i in range(num_windows_x):
                for j in range(num_windows_y):
                    window_left = i * window_size
                    window_top = j * window_size
                    window_right = min((i + 1) * window_size, width)
                    window_bottom = min((j + 1) * window_size, height)
                    window = rasterio.windows.Window(window_left, window_top, window_right - window_left, window_bottom - window_top)
                    data = dataset.read(window=window)
                    pil_image = Image.fromarray(np.transpose(data, (1, 2, 0)))
                    masks, _, _, _= sam.generate(source=img_path, batch=Tr)
                    mask_overlay = np.zeros_like(data[0])
                    print(mask_overlay.shape)
                    for i, mask in enumerate(masks):
                        mask = mask.cpu().numpy().astype('uint8')
                        # print(i, mask)
                        mask_overlay += ((mask > 0) * (i + 1)).astype('uint8')
                    print(mask_overlay.shape)
                    dst.write(mask_overlay, window=window, indexes=1)
                    # Sử dụng dữ liệu trong window tại đây
                # print(f"Window ({i}, {j}):", data.shape)

# Sử dụng hàm để đọc và truy cập từng phần của ảnh
# filename = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/POC_AOI_Here_VN_Final_Image_10CM.tif"
#   # Kích thước cửa sổ
# read_image_in_windows(filename, window_size)



text_prompt = "rooftop"
image = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/a_4326.tif"
fp_out_img = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/A_Test_xin/a_4326_mask.tif"
# fp_out_shp = r'/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/data_demo/rooftop1.shp'
window_size = 1000
chkpnt_dir = r"/home/skm/SKM/WORK/ALL_CODE/WorkSpace_SKM_DucAnh/SAM/model/"
checkpoint = os.path.join(chkpnt_dir, "sam_vit_h_4b8939"+".pth")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sam = LangSAM()
read_image_in_windows(image, window_size, fp_out_img)

# sam.predict(image, text_prompt, box_threshold=0.24, text_threshold=0.24, return_results=True, output=fp_out_img)
# sam.raster_to_vector(fp_out_img, fp_out_shp)
