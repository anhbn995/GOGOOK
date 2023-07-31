import numpy as np
import gdal
import os
import shutil
import argparse
import sys

def main(img_dir, mask_dir, thres):
    files_list = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    negative_fn = []
    for fn in files_list:
        dataset = gdal.Open(os.path.join(mask_dir,fn))
        values = dataset.ReadAsArray()        
        h,w = values.shape
        print(round(thres*h*w))
        if np.count_nonzero(values) <= thres*h*w:
            negative_fn.append(fn)
    count = len(negative_fn)
    print("Black mask count: " + str(count) + "/" +str(len(files_list)))
    remove = int(input("Enter number of black sample to remove:" ))
    rm = max(0,min(remove, count))
    for i in range(rm):
        remove_file(img_dir, negative_fn[i])
        remove_file(mask_dir, negative_fn[i])

        
def remove_file(imagedir, file_name):
    os.remove(os.path.join(imagedir,file_name))


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()

    args_parser.add_argument(
        '--img_dir',
        help='Orginal Image Directory',
        required=True
    )


    args_parser.add_argument(
        '--mask_dir',
        help='Mask directory',
        required=True
    )

    args_parser.add_argument(
        '--thres',
        help='1 pixel thres',
        type=float,
        default=0.95
    )

    param = args_parser.parse_args()
    img_dir= param.img_dir
    mask_dir= param.mask_dir
    thres=param.thres
    
    main(img_dir, mask_dir, thres)



