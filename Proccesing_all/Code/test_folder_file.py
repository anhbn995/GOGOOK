
import os,glob
name_folder_dir = "/media/skm/Image/SOI_Drone_1/Image/image32644_resize"
# name_folder_out = ""
os.chdir(name_folder_dir)
for file_ in glob.glob("*.tif"):
    print(file_[:-4])