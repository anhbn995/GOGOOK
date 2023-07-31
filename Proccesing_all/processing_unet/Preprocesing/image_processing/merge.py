import glob
import os
import sys
sys.path.append(r'C:\Program Files\QGIS 2.18\bin')
import gdal_merge as gm
input_dir = os.path.abspath(sys.argv[1])
output_dir = os.path.abspath(sys.argv[2])
def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file)
        # print(file[:-4])
    return list_id
file_list = create_list_id(input_dir)
list_string = ['', '-o']
output_file = os.path.join(output_dir,'merged.tif')
list_string.append(str(output_file))
for file_name in file_list:
    list_string.append(str(os.path.join(input_dir,file_name)))
gm.main(list_string)

# files_string = " ".join(file_list)
# command = "gdal_merge.py -o output.tif -of gtiff " + files_string
# os.system(command)