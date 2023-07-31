import glob
import os
import sys
from subprocess import call
# sys.path.append(r'C:\Program Files\QGIS 2.18\bin')
# import gdal_merge as gm
input_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
output_nane = str(sys.argv[3])
def create_list_id(path):
    list_image = []
    # files = os.listdir(path)
    # for dir_name in files:
    #     for 
    # return files
    for root, dirs, files in os.walk(path):
        # print(dirs)
        for file in files:
            if file.endswith(".tif"): #and 'vh' in file.lower():
                list_image.append(os.path.join(root, file))
    return list_image
if __name__ == "__main__": 
    file_list = create_list_id(input_dir)
    file_list.sort()
    with open(os.path.join(output_dir,'{}.txt'.format(output_nane)), 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)
    print(file_list)
    list_string = ['gdal_merge.py','-of','gtiff','-o']
    output_file = os.path.join(output_dir,'{}.tif'.format(output_nane))
    print(output_file)
    list_string.append(str(output_file))
    list_string.append("-separate")
    for file_name in file_list:
        list_string.append(file_name)
    call(list_string)
# gm.main(list_string)

# files_string = " ".join(file_list)
# command = "gdal_merge.py -o output.tif -of gtiff " + files_string
# os.system(command)
