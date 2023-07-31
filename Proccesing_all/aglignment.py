from osgeo import gdal, osr, ogr
from gdalconst import GA_ReadOnly
from subprocess import call
import numpy as np
import os
import sys
class ImageAlignment:
    def __init__(self, basefile, imagefile, fileprefix):
        self.basefile=basefile
        self.imagefile=imagefile
        self.fileprefix=fileprefix

    def align(self):
        data = gdal.Open(self.basefile, GA_ReadOnly)
        geoTransform = data.GetGeoTransform()
        resx = geoTransform[1]
        resy = -geoTransform[5]

        s=['gdal_translate', '-tr', '{}'.format(resx),'{}'.format(resy),'-of', 'GTiff', '-ot', 'Byte', '-a_nodata', '0','{}'.format(self.imagefile),'{}/image.tif'.format(self.fileprefix)]
        call(s)

        data=None

    def clip(self):
        reffile='{}/image.tif'.format(self.fileprefix)
        data = gdal.Open(reffile, GA_ReadOnly)
        geoTransform = data.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * data.RasterXSize
        miny = maxy + geoTransform[5] * data.RasterYSize
        resx = geoTransform[1]
        resy = -geoTransform[5]     
        #s='gdalwarp -tr {} {}'.format(resx, resy) + ' -tap -of GTiff -ot Byte -srcnodata 0 -dstnodata 0 ' + self.basefile + ' {}/tmp.tif'.format(self.fileprefix)
        #call(s, shell=True)
        s=['gdal_translate','-projwin','{}'.format(minx),'{}'.format(maxy),'{}'.format(maxx),'{}'.format(miny),'-of', 'GTiff', '-ot', 'Byte', '{}'.format(self.basefile), '{}/base.tif'.format(self.fileprefix)]
        call(s)

def image_align(basefile,imagefile,fileprefix):
    al=ImageAlignment(basefile, imagefile, fileprefix)
    al.align()
    al.clip()
    al.settranparencearea()
    return True
if __name__=='__main__':
    basefile=sys.argv[1]
    imagefile=sys.argv[2]
    fileprefix=sys.argv[3]
    al=ImageAlignment(basefile, imagefile, fileprefix)
    al.align()
    al.clip()
