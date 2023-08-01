import rasterio
import matplotlib.pyplot as plt

path_img = r"C:\Users\SkyMap\Desktop\a.tif"
raster = rasterio.open(path_img)
print(type(raster))