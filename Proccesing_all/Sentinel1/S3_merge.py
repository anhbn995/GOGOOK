import glob
import os
import sys
from subprocess import call
import time
x =  time.time()
# sys.path.append(r'C:\Program Files\QGIS 2.18\bin')
# import gdal_merge as gm
input_dir = os.path.abspath(sys.argv[1])

output_dir = input_dir + '_stack'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

file_list = create_list_id(input_dir)
print(len(file_list))

file_list.sort()
# file_list.reverse()
list_string = ['gdal_merge.py','-of','gtiff','-o']
output_file = os.path.join(output_dir,'stack.tif')
list_string.append(str(output_file))
list_string.append("-separate")
for file_name in file_list:
    list_string.append(file_name)
call(list_string)
y = time.time()
print("Het: {} ph".format((y-x)/60))
# gm.main(list_string)

# files_string = " ".join(file_list)
# command = "gdal_merge.py -o output.tif -of gtiff " + files_string
# os.system(command)
