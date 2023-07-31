from osgeo import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import os
import random
from tqdm import *
class CD_GenerateTrainingDataset:
    def __init__(self, basefile, imagefile, labelfile, sampleSize, outputFolder, fileprefix):
        # self.basefile=basefile
        self.imagefile=imagefile
        self.labelfile=labelfile
        self.sampleSize=sampleSize
        self.outputFolder=outputFolder
        self.fileprefix=fileprefix
        self.outputFolder_base=None
        self.outputFolder_image=None

    def generateTrainingDataset(self, nSamples):
        self.outputFolder_base = os.path.join(self.outputFolder,"base")
        self.outputFolder_image = os.path.join(self.outputFolder,"image")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        if not os.path.exists(self.outputFolder_base):
            os.makedirs(self.outputFolder_base)
        if not os.path.exists(self.outputFolder_image):
            os.makedirs(self.outputFolder_image)
        # base=gdal.Open(self.basefile, GA_ReadOnly)
        # basedata=np.array(base.ReadAsArray())

        image=gdal.Open(self.imagefile, GA_ReadOnly)
        # imagedata=np.array(image.ReadAsArray())

        raster = gdal.Open(self.labelfile, GA_ReadOnly)
        geo = raster.GetGeoTransform()
        proj=raster.GetProjectionRef()
        size_X=raster.RasterXSize
        size_Y=raster.RasterYSize

        rband=np.array(raster.GetRasterBand(1).ReadAsArray())

        icount=0
        with tqdm(total=nSamples) as pbar:
            while icount<nSamples:
                px=random.randint(0,size_X-1-self.sampleSize)
                py=random.randint(0,size_Y-1-self.sampleSize)
                rband = raster.GetRasterBand(1).ReadAsArray(px, py, self.sampleSize, self.sampleSize)
                # lable=rband[py:py+self.sampleSize, px:px+self.sampleSize]
                if np.amax(rband)>0:
                    geo1=list(geo)
                    geo1[0]=geo[0]+geo[1]*px
                    geo1[3]=geo[3]+geo[5]*py
                    geo1=tuple(geo1)
                    # labelfile=os.path.join(self.outputFolder, self.fileprefix+'{:03d}_label.tif'.format(icount+1))
                    # self.writeLabelAsFile(lable, labelfile, geo1, proj)

                    # basefile=os.path.join(self.outputFolder_base, self.fileprefix+'{:03d}.tif'.format(icount+1))
                    # gdal.Translate(basefile, base,srcWin = [px,py,self.sampleSize,self.sampleSize])
                    # self.writeDataAsFile(bdata, basefile, geo1, proj)
                    imagefile=os.path.join(self.outputFolder_image, self.fileprefix+'{:03d}.tif'.format(icount+1))
                    gdal.Translate(imagefile, image,srcWin = [px,py,self.sampleSize,self.sampleSize])
                    # self.writeDataAsFile(idata, imagefile, geo1, proj)

                    icount+=1
                    # print(icount)
                    pbar.update()


        raster=None
        image=None
        base=None

    def writeLabelAsFile(self, data, filename, geo, proj):
        size_Y, size_X=data.shape
        # Create tiff file
        target_ds = gdal.GetDriverByName('GTiff').Create(filename, size_X, size_Y, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo)
        target_ds.SetProjection(proj)
        band = target_ds.GetRasterBand(1)
        target_ds.GetRasterBand(1).SetNoDataValue(0)				
        band.WriteArray(data)
        band.FlushCache()

        target_ds=None
        
    def writeDataAsFile(self, data, filename, geo, proj):
        nbands, size_Y, size_X=data.shape
        # Create tiff file
        target_ds = gdal.GetDriverByName('GTiff').Create(filename, size_X, size_Y, nbands, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo)
        target_ds.SetProjection(proj)
        for i in range(0, nbands):
            band = target_ds.GetRasterBand(i+1)
            band.SetNoDataValue(0)	
            band.WriteArray(data[i,:,:])
            band.FlushCache()

        target_ds=None        

if __name__=='__main__':
    basefile='/media/skm/Image/SotXuatHuyet/23Nov2019/Image/31_29_Ward_1_1.tif'
    imagefile='/media/skm/Image/SotXuatHuyet/23Nov2019/Image/31_29_Ward_1_1.tif'
    labelfile='/media/skm/Image/SotXuatHuyet/23Nov2019/Image_mask/31_29_Ward_1_1.tif'
    sampleSize=512
    outputFolder='/media/skm/Image/SotXuatHuyet/23Nov2019/Training_dataset/buffer_nho_hon'
    fileprefix='sla'

    gen=CD_GenerateTrainingDataset(basefile, imagefile, labelfile, sampleSize, outputFolder, fileprefix)

    gen.generateTrainingDataset(2000)