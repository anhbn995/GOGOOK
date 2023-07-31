import rasterio.mask
import rasterio
import numpy as np
import glob, os
from tqdm import tqdm

label = "/mnt/Nam/bairac/classification_data/xxx/"
if not os.path.exists(label):
    os.mkdir(label)
for img in tqdm(glob.glob("/mnt/Nam/bairac/classification_data/landfill_training/*.tif")):
    with rasterio.open(img, mode='r+') as src:
        out_meta=src.meta
        out_meta.update({'nodata': 0, 'dtype': 'float32'})
        band=src.read().astype(float)
        for i in range(src.count):
            band[i][band[i]==0.]=np.nan
            band[i][band[i]==65536.0]=np.nan
            minn = np.nanpercentile(band[i], 2)
            maxx = np.nanpercentile(band[i], 98)
            band[i] = np.interp(band[i], (minn, maxx), (0, 1))
            band[i][np.isnan(band[i])]=0.0
        with rasterio.open(label+os.path.basename(img) , 'w', compress='RAW', **out_meta) as ras:
            ras.write(band.astype('float32'))
    break