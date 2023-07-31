import gdal
import numpy as np
import random

class DataGenerator:
    def __init__(self, img, mask=None, train=True):
        self.img=img
        self.mask=mask
        self.train =train
        self.ds = gdal.Open(self.img, gdal.GA_ReadOnly)
        self.bcount=self.ds.RasterCount
        self.rows = self.ds.RasterXSize
        self.cols = self.ds.RasterYSize
        
        self.bands=[]
        for i in range (self.bcount):
            band = np.array(self.ds.GetRasterBand(i+1).ReadAsArray())
            self.bands.append(band)
            band=None

        self.dsmask = gdal.Open(self.mask, gdal.GA_ReadOnly)
        self.bmaskcount=self.dsmask.RasterCount
        self.bandmask=[]
        for i in range (self.bmaskcount):
            band = np.array(self.dsmask.GetRasterBand(i+1).ReadAsArray())
            self.bandmask.append(band)
            band=None

        random.seed(123)
        

    def __del__(self):
        self.ds=None
        self.bands=None
        self.dsmask=None
    
    def generate(self):
        print("Generator...")
        size=128
        while (True):
            i=random.randrange(self.cols-size)
            j=random.randrange(self.rows-size)
            x=np.zeros((size, size, self.bcount))
            y=np.zeros((size, size, self.bmaskcount))
            for h in range(self.bcount):
                x[:,:,h]=self.bands[h][i:i+size, j:j+size]

            for h in range(self.bmaskcount):
                y[:,:,h]=self.bandmask[h][i:i+size, j:j+size]

            yield (x, y)


if __name__=='__main__':
    img=r"F:\data2019\Thailand\sugarcane\AngThong_byte.tif"
    mask=r"F:\data2019\Thailand\sugarcane\AngThong_mask.tif"
    train=True
    tool=DataGenerator(img, mask, train)
    gen=tool.generate()
    for i in range(4000):
        (x,y)=next(gen)
        np.save("./train/images/img{}.npy".format(i,), x)
        np.save("./train/masks/img{}.npy".format(i,), y)

    # print (x,y)
    tool=None        