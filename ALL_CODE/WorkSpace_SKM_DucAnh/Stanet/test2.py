import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import random
from PIL import Image,ImageFilter

def __blur(img):
    # if img.mode == 'RGB':
    img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))
    return img

img = Image.open(r'/home/skm/SKM16/Tmp/Stanet4band/SplitTrainValTest/test/A/img_stack_0_188.tif')#.convert('RGB')
# transform = transforms.Grayscale(1)
# osize = [128, 128]
# method=Image.BICUBIC
transform = transforms.Lambda(lambda img: __blur(img))
img = transform(img)
# np.array(img).shape
# img.show()
# img.filter(ImageFilter.GaussianBlur(radius=random.random()))
img.show()