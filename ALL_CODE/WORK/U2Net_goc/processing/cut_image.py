import rasterio.mask
import rasterio
from rasterio import windows
from itertools import product
import glob, os

# path_image = '/media/skymap/Learnning/public/farm-bing18/Data/crop_data/'
path_img = '/media/skymap/Learnning/public/farm-bing18/farm_malaysia/data/Kunak.tif'
output_box = 'Kunak_{}_{}.tif'
out_path_image = '/media/skymap/Learnning/public/farm-bing18/farm_malaysia/data_cut/'

# xxx = []
# for i in glob.glob(path_image):
#     with rasterio.open(i) as rr:
#         xxx.append(rr.transform)

# listtt = ['box_57.tif', 'box_46.tif', 'box_49.tif', 'box_50.tif', 'box_45.tif']
# xxx = []
# for i in listtt:
#     with rasterio.open(path_image+i) as rr:
#         print(rr.crs.to_string())
#         xxx.append(rr.transform)


def get_tiles(ds, width, height):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, 19500), range(0, nrows, 19500))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in offsets: 
        window=windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform


with rasterio.open(path_img) as inds:
    tile_width, tile_height = 20000, 20000
    projstr = inds.crs.to_string()
    height = inds.height
    width = inds.width
    print(height, width)
    out_meta = inds.meta
    transformss = inds.transform
    
    for window, transform in get_tiles(inds, tile_width, tile_height):
        outpath_image = os.path.join(out_path_image, output_box.format(window.col_off, window.row_off))
        out_image = inds.read(window=window)
        out_meta.update({"driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": transform})
        with rasterio.open(outpath_image, "w", compress='lzw', **out_meta) as dest:
            dest.write(out_image)