import rasterio 

path1 = r"/home/geoai/geoai_data_test/data_npark/Sentinel/classification/2022_T1.tif"
path2 = r"/home/geoai/geoai_data_test/data_npark/Sentinel/classification/2021_T10.tif"

src = rasterio.open(path2)
meta = src.meta
with rasterio.open(path1) as s:
    img = s.read()

with rasterio.open(path1, 'w', **meta) as dst:
    dst.write(img)
    dst.write_colormap(
    1, {
        0: (0, 0, 0, 0),
        1: (34,139,34,255),
        2: (100, 149, 237, 255),
        3: (101,67,33, 255)
        })