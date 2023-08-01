import cv2
import numpy as np
import matplotlib.pyplot as plt
import rasterio

def generate_blank_image(fp_img):
    mask = cv2.imread(fp_img)
    # mask = np.ones(shape=(512, 512, 3), dtype=np.int16)
    return mask, int(mask.shape[0]/2)

fp = r"C:\Users\SkyMap\Desktop\AAA.tif"
sizes = [0.5, 1, 2]
# sizes = [0.8, 1, 1.25]
# anchors = (16, 32, 64, 128, 256)
anchors = (40, 80, 160, 320)
img1, centroid = generate_blank_image(fp)
print(centroid)

for anchor in anchors:
    for size in sizes:
        h = anchor*size
        w = anchor/size
        print(h,w)
        pt1 = (int(centroid + w/2), int(centroid + h/2))
        pt2 = (int(centroid - w/2), int(centroid - h/2))
        print(pt1, pt2)
        print('........')
        cv2.rectangle(img1, pt1=pt1, pt2=pt2, color=(255,0,0), thickness=2)

plt.imshow(img1)