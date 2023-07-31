from osgeo import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import os
import random
import cv2
class CD_GenerateTrainingDataset:
    def __init__(self, basefile, imagefile, labelfile, sampleSize, outputFolder, fileprefix):
        self.basefile=basefile
        self.imagefile=imagefile
        self.labelfile=labelfile
        self.sampleSize=sampleSize
        self.outputFolder=outputFolder
        self.fileprefix=fileprefix

    def generateTrainingDataset(self, nSamples):
        base=gdal.Open(self.basefile, GA_ReadOnly)
        basedata=np.array(base.ReadAsArray())

        image=gdal.Open(self.imagefile, GA_ReadOnly)
        imagedata=np.array(image.ReadAsArray())

        raster = gdal.Open(self.labelfile, GA_ReadOnly)
        geo = raster.GetGeoTransform()
        proj=raster.GetProjectionRef()
        size_X=raster.RasterXSize
        size_Y=raster.RasterYSize

        rband=np.array(raster.GetRasterBand(1).ReadAsArray())
        rband=cv2.bitwise_or(rband,raster.GetRasterBand(2).ReadAsArray())
        rband=cv2.bitwise_or(rband,raster.GetRasterBand(3).ReadAsArray())

        icount=0
        while icount<nSamples:
            px=random.randint(0,size_X-1-self.sampleSize)
            py=random.randint(0,size_Y-1-self.sampleSize)
            lable=rband[py:py+self.sampleSize, px:px+self.sampleSize]
            if np.amax(lable)>0:
                geo1=list(geo)
                geo1[0]=geo[0]+geo[1]*px
                geo1[3]=geo[3]+geo[5]*py
                geo1=tuple(geo1)
                # labelfile=os.path.join(self.outputFolder, self.fileprefix+'{:03d}_label.tif'.format(icount+1))
                # self.writeLabelAsFile(lable, labelfile, geo1, proj)

                basefile=os.path.join(self.outputFolder, self.fileprefix+'{:03d}_image.tif'.format(icount+1))
                bdata=basedata[:,py:py+self.sampleSize, px:px+self.sampleSize]
                self.writeDataAsFile(bdata, basefile, geo1, proj)
                imagefile=os.path.join(self.outputFolder, self.fileprefix+'{:03d}_label.tif'.format(icount+1))
                idata=imagedata[:,py:py+self.sampleSize, px:px+self.sampleSize]
                self.writeData2AsFile(idata, imagefile, geo1, proj)

                icount+=1
                print(icount)


        raster=None

    def writeLabelAsFile(self, data, filename, geo, proj):
        size_Y, size_X=data.shape
        # Create tiff file
        target_ds = gdal.GetDriverByName('GTiff').Create(filename, size_X, size_Y, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo)
        target_ds.SetProjection(proj)
        band = target_ds.GetRasterBand(1)
        # target_ds.GetRasterBand(1).SetNoDataValue(0)				
        band.WriteArray(data)
        band.FlushCache()

        target_ds=None

    def writeData2AsFile(self, data, filename, geo, proj):
        nbands, size_Y, size_X=data.shape
        # Create tiff file
        target_ds = gdal.GetDriverByName('GTiff').Create(filename, size_X, size_Y, nbands, gdal.GDT_Byte)
        target_ds.SetGeoTransform(geo)
        target_ds.SetProjection(proj)
        for i in range(0, nbands):
            band = target_ds.GetRasterBand(i+1)
            # band.SetNoDataValue(0)	
            band.WriteArray(data[i,:,:])
            band.FlushCache()

        target_ds=None               
    def writeDataAsFile(self, data, filename, geo, proj):
        
        nbands, size_Y, size_X=data.shape
        # Create tiff file
        target_ds = gdal.GetDriverByName('GTiff').Create(filename, size_X, size_Y, nbands, gdal.GDT_Float32)
        target_ds.SetGeoTransform(geo)
        target_ds.SetProjection(proj)
        for i in range(0, nbands):
            band = target_ds.GetRasterBand(i+1)
            # band.SetNoDataValue(0)	
            band.WriteArray(data[i,:,:])
            band.FlushCache()

        target_ds=None        

if __name__=='__main__':
    basefile=r"F:\Agricultrure\Thailand\Data\img\img.tif"
    imagefile=r"F:\Agricultrure\Thailand\Data\image3band_backbone\image.tif"
    labelfile=r"F:\Agricultrure\Thailand\Data\image3band_backbone\image.tif"
    sampleSize=256
    outputFolder=r"F:\Agricultrure\Thailand\Data\data_Train_newmodel"
    fileprefix='image_'

    gen=CD_GenerateTrainingDataset(basefile, imagefile, labelfile, sampleSize, outputFolder, fileprefix)

    gen.generateTrainingDataset(2000)
