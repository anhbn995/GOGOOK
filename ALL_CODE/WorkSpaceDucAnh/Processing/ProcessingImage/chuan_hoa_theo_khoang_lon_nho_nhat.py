import os, glob
import rasterio
import numpy as np


dir_img_height = r"/home/skm/SKM16/Planet_GreenChange/Data_4band_Green/ImgRGBNir"
dir_img_height_255 = r"/home/skm/SKM16/Planet_GreenChange/Data_4band_Green/ImgRGBNir_255_gdal"
os.makedirs(dir_img_height_255, exist_ok=True)

list_image = glob.glob(os.path.join(dir_img_height, "*.tif"))    
Tmin = 1000000000
Tmax = -1000000000
for fp_height in list_image:
    # print(fp_height)
    with rasterio.open(fp_height) as src:
        arr = src.read()
    if(np.amin(arr) < Tmin):
        Tmin = np.amin(arr)
    if(np.amax(arr) > Tmax):
        Tmax = np.amax(arr)

data_Tmin_Tmax = f"Tmax:{Tmax} \n Tmin:{Tmin}"
file_save_Tmin_Tmax = os.path.join(dir_img_height_255, 'Tmin_Tmax.txt')
with open(file_save_Tmin_Tmax, "w") as file:
    file.write(data_Tmin_Tmax)

for fp_in in list_image:
    fp_out = fp_in.replace(dir_img_height, dir_img_height_255)
    print("gdal_translate -scale " + str(Tmin) + " " + str(Tmax) + " 0 255 " + fp_in + " " + fp_out)
    os.system("gdal_translate -scale " + str(Tmin) + " " + str(Tmax) + " 0 255 " + fp_in + " " + fp_out)