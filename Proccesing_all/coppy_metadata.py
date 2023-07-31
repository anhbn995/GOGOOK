import rasterio


# img_dest = r"/home/geoai/geoai_data_test/data_npark/Sentinel/classification/2021_T1.tif"
# img_change = r"/home/geoai/geoai_data_test/data_npark/Sentinel/classification/2021_T12.tif"

# src = rasterio.open(img_dest)
# profile = src.meta

# with rasterio.open(img_change) as s:
#     img = s.read()

# with rasterio.open(img_change, 'w', **profile) as dest: 
#     dest.write(img)
#     dest.write_colormap(
#     1, {
#         0: (0, 0, 0, 0),
#         1: (34,139,34,255),
#         2: (100, 149, 237, 255),
#         3: (101,67,33, 255)
#         })


img_dest = r"/home/geoai/geoai_data_test/data_npark/Sentinel/mosaic/2021_T1.tif"
img_change = r"/home/geoai/geoai_data_test/data_npark/Sentinel/mosaic/2021_T12.tif"

src = rasterio.open(img_dest)
profile = src.meta

with rasterio.open(img_change) as s:
    img = s.read()

with rasterio.open(img_change, 'w', **profile) as dest: 
    dest.write(img)
