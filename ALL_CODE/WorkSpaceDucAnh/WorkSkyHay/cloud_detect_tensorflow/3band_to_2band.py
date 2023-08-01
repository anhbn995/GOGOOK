from multiprocessing.pool import Pool
import rasterio
import numpy as np
from tqdm import *

if __name__ == '__main__':
    im_path = r"./raw_more/cloud/sentinel_1 (2).tif"
    mask_path = r"./raw_more/shadow/sentinel_1 (2)_label.tif"
    out_path = r"./raw_more/cloud/sentinel_1 (2)_label.tif"
    with rasterio.open(mask_path) as msk:
        with rasterio.open(im_path) as src:
            out_meta = src.meta
            # print(out_meta)
        src.close()
        out_meta.update({"count": 1})
        mask = msk.read()
        mask[mask==3] = 1
        with rasterio.open(out_path, "w", compress='lzw', **out_meta) as dest:
            dest.write(mask.astype(np.uint8))
        dest.close()
    msk.close()



