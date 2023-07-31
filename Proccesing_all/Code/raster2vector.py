from osgeo import gdal, ogr
import sys
import os


gdal.UseExceptions()
os.chdir("/media/skm/SKM/Czech/cezch_new/Img/predict_unit8/20200519_164432_val_weights_last")
fileName = "/media/skm/SKM/Czech/cezch_new/Img/predict_unit8/20200519_164432_val_weights_last/decidous19LESYM3_003_0021_rgb_epsg32633.tif"
src_ds = gdal.Open(fileName)
if src_ds is None:
    print('Unable to open %s' % src_fileName)
    sys.exit(1)
srcband = src_ds.GetRasterBand(1)
dst_layername = "PolyFtr"
drv = ogr.GetDriverByName("ESRI Shapefile")
dst_ds = drv.CreateDataSource(dst_layername + ".shp")
dst_layer = dst_ds.CreateLayer(dst_layername, srs = None)
newField = ogr.FieldDefn('Area', ogr.OFTInteger)
dst_layer.CreateField(newField)
gdal.Polygonize(srcband, None, dst_layer, 0, [], 
callback=None )