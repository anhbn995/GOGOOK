import rasterio
import numpy as np
def read_mask(url):
    with rasterio.open(url) as src:
        array = src.read()[0]
    return (array > 127).astype(np.uint8)*255

def save_img(img_path,mask_path,outputFileName):

    mask = read_mask(mask_path)
    with rasterio.open(img_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height

    crs = rasterio.crs.CRS({"init": "epsg:4326"})
    new_dataset = rasterio.open(outputFileName, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
# print(masking(r"/media/building/building/data_source/tmp/Malaysia-jupem/image_mask/forest.tif")[0].shape)
    new_dataset.write(mask,1)
    new_dataset.close()
if __name__ == '__main__':
    save_img(r"/media/khoi/Data1/Gap/forest/img/bhuvan11.tif",r"/media/khoi/Data1/Gap/bhuvan11.png",r"/media/khoi/Data1/Gap/bhuvan11_road.tif")

