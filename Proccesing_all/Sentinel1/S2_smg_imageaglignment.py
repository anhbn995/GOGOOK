from osgeo import gdal, osr, ogr
from gdalconst import GA_ReadOnly
from subprocess import call
import numpy as np
import os
import sys
import shutil
def create_list_id(path):
    list_image = []
    # files = os.listdir(path)
    # for dir_name in files:
    #     for 
    # return files
    for root, dirs, files in os.walk(path):
        print(dir)
        for file in files:
            if file.endswith(".tif"):
                list_image.append(os.path.join(root, file))
    return list_image
class ImageAlignment:
    def __init__(self,foder_path, list_image):
        self.basefile=list_image[0]
        self.fileprefix=foder_path+'_align'
        self.foder_path = foder_path
        self.list_image = list_image
        if not os.path.exists(self.fileprefix):
                os.makedirs(self.fileprefix)
        for image_path1 in list_image:
            dir_name = os.path.basename(os.path.dirname(image_path1))
            if not os.path.exists(os.path.join(self.fileprefix,dir_name)):
                os.makedirs(os.path.join(self.fileprefix,dir_name))
    def align(self):
        image_base_name = os.path.basename(self.basefile)
        dir_base_name = os.path.basename(os.path.dirname(self.basefile))
        path_out_base = os.path.join(self.fileprefix,dir_base_name,image_base_name)
        shutil.copyfile(self.basefile, path_out_base)


        self.data = gdal.Open(self.basefile, GA_ReadOnly)
        data = self.data
        geoTransform = data.GetGeoTransform()
        resx = geoTransform[1]
        resy = -geoTransform[5]
        
        for image_path in self.list_image[1:]:
            image_name = os.path.basename(image_path)[:-4]
            dir_name = os.path.basename(os.path.dirname(image_path))
            path_out_tmp = os.path.join(self.fileprefix,dir_name,image_name+"_tmp"+".tif")
            s=['gdal_translate', '-tr', '{}'.format(resx),'{}'.format(resy),'-of', 'GTiff', '-ot', 'Byte', '-a_nodata', '0','-colorinterp_4','undefined',image_path,path_out_tmp]
            call(s)

        data=None

    def clip(self):
        data = gdal.Open(self.basefile)
        # data = self.data
        geoTransform = data.GetGeoTransform()
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * data.RasterXSize
        miny = maxy + geoTransform[5] * data.RasterYSize
        resx = geoTransform[1]
        resy = -geoTransform[5]     
        #s='gdalwarp -tr {} {}'.format(resx, resy) + ' -tap -of GTiff -ot Byte -srcnodata 0 -dstnodata 0 ' + self.basefile + ' {}/tmp.tif'.format(self.fileprefix)
        #call(s, shell=True)
        for image_path in self.list_image[1:]:
            image_name = os.path.basename(image_path)[:-4]
            dir_name = os.path.basename(os.path.dirname(image_path))
            path_out_tmp = os.path.join(self.fileprefix,dir_name,image_name+"_tmp"+".tif")
            path_out_image = os.path.join(self.fileprefix,dir_name,image_name+".tif")
            s=['gdal_translate','-projwin','{}'.format(minx),'{}'.format(maxy),'{}'.format(maxx),'{}'.format(miny),'-of', 'GTiff', '-ot', 'Byte','-a_nodata', '0','-colorinterp_4','undefined', '{}'.format(path_out_tmp), '{}'.format(path_out_image)]
            call(s)
            os.remove(path_out_tmp)

def image_align(basefile,imagefile,fileprefix):
    al=ImageAlignment(basefile, imagefile, fileprefix)
    al.align()
    al.clip()
    al.settranparencearea()
    return True
if __name__=='__main__':
    foder_path = sys.argv[1]
    list_image = create_list_id(foder_path)
    al=ImageAlignment(foder_path,list_image)
    al.align()
    al.clip()
