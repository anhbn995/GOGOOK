import rasterio
from rasterio.crs import CRS

# Đọc thông tin của 2 tệp ảnh
fp1 = r'/home/skm/SKM16/X/Test/img_origin (copy)/20181031.tif'
with rasterio.open(r'/home/skm/SKM16/X/Test/img_origin (copy)/20181031.tif') as src1:
# with rasterio.open(r'/home/skm/SKM16/Tmp/npark-backend-v2/data_projects/Green Cover Npark Singapore/base/2022_T3.tif') as src1:
    transform1 = src1.transform
    crs1 = src1.crs
    bounds1 = src1.bounds
    
fp2 = r'/home/skm/SKM16/X/Test/Landslide_Sentinel-2_DAnh/DaBac/Slope_10m.tif'
with rasterio.open(r'/home/skm/SKM16/X/Test/Landslide_Sentinel-2_DAnh/DaBac/Slope_10m.tif') as src2:
# with rasterio.open(r'/home/skm/SKM16/Tmp/npark-backend-v2/data_projects/Green Cover Npark Singapore/base/2022_T3.tif') as src2:
    transform2 = src2.transform
    crs2 = src2.crs
    bounds2 = src2.bounds

# Tìm vùng chồng lấp của hai tệp ảnh
xmin = max(bounds1.left, bounds2.left)
ymin = max(bounds1.bottom, bounds2.bottom)
xmax = min(bounds1.right, bounds2.right)
ymax = min(bounds1.top, bounds2.top)

# Chuyển đổi vùng chồng lấp sang tọa độ của tệp ảnh thứ hai
from rasterio.warp import transform_bounds
# print(transform1)
leftw, bottomw, rightw, topw = transform_bounds(crs1, crs2, xmin, ymin, xmax, ymax)
left, bottom, right, top = transform_bounds(crs2, crs1, leftw, bottomw, rightw, topw)


from rasterio.windows import Window
from rasterio.transform import from_bounds

print("Vùng chồng lấp của hai tệp ảnh là: ", left, bottom, right, top)
# Tính toán kích thước của vùng cần cắt trên ảnh
bbox = (left, bottom, right, top)

def clip_raster_by_bbox(input_path, output_path, bbox):
    with rasterio.open(input_path) as src:
        # Get the window to read from
        minx, miny, maxx, maxy = bbox
        window = src.window(minx, miny, maxx, maxy)

        # Calculate the width and height of the output
        width = window.width
        height = window.height

        # Compute the transform for the output
        transform = rasterio.windows.transform(window, src.transform)

        # Update the metadata for the output
        meta = src.meta.copy()
        meta.update({
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'transform': transform
        })

        # Read and write the data
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(src.read(window=window))
clip_raster_by_bbox(fp2, fp2.replace('.tif','_okkk.tif'), bbox)

# with rasterio.open(r'/home/skm/SKM16/X/Test/img_origin (copy)/20181031.tif') as src:
#     row_start, col_start = map(int, ~transform1 * (left, top))
#     row_stop, col_stop = map(int, ~transform1 * (right, bottom))
    
#     win = ((row_start, row_stop), (col_start, col_stop))
#     # data, mask = src.read(window=win)
#     profile = src.profile
#     profile.update(width=row_stop - row_start, height=col_stop - col_start, transform=rasterio.windows.transform(win, src.transform))
#     with rasterio.open(r'/home/skm/SKM16/X/Test/img_origin (copy)/a1.tif', 'w', **profile) as dst:
#         dst.write(src.read(window=win))

# with rasterio.open(r'/home/skm/SKM16/X/Test/Landslide_Sentinel-2_DAnh/DaBac/Slope_10m.tif') as src:
#     row_start, col_start = map(int, ~transform2 * (left, top))
#     row_stop, col_stop = map(int, ~transform2 * (right, bottom))
#     win = ((row_start, row_stop), (col_start, col_stop))
#     # data, mask = src.read(window=win)
#     profile = src.profile
#     profile.update(width=row_stop - row_start, height=col_stop - col_start, transform=rasterio.windows.transform(win, src.transform))
#     with rasterio.open(r'/home/skm/SKM16/X/Test/img_origin (copy)/c1.tif', 'w', **profile) as dst:
#         dst.write(src.read(window=win))