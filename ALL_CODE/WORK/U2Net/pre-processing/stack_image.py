import os
from subprocess import call

# input_name1 = "/mnt/data/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A  Dsc 12-February-2021.tif"
# input_name1 = "/mnt/data/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A  Dsc 24-February-2021.tif"
input_name1 = "/mnt/data/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A Dsc 10-Oct-2021.tif"
input_name2 = "/mnt/data/public/changedetection_SAR/pipeline/Raw/Costa Rica S1A Dsc 22-Oct-2021.tif"

input_name3 = "/mnt/data/public/changedetection_SAR/newdataFeb-2020/Costa Rica S1A  Dsc 06-February-2020.tif"
input_name4 = "/mnt/data/public/changedetection_SAR/newdataFeb-2020/Costa Rica S1A  Dsc 18x-February-2020.tif"

output_dir = "/mnt/data/public/changedetection_SAR/pipeline/nammmm/stack_data/"
output_nane = "stack_4img_Oct21_feb20"


if __name__ == "__main__":
    file_list = [input_name1, input_name2, input_name3, input_name4]
    with open(os.path.join(output_dir,'{}.txt'.format(output_nane)), 'w') as f:
        for item in file_list:
            f.write("%s\n" % item)
    list_string = ['gdal_merge.py','-of','gtiff','-o']
    output_file = os.path.join(output_dir,'{}.tif'.format(output_nane))
    print(output_file)
    list_string.append(str(output_file))
    list_string.append("-separate")
    for file_name in file_list:
        list_string.append(file_name)
    call(list_string)