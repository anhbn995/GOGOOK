import sys
import glob, os
import numpy as np


def create_list_id(path):
    list_id = []
    os.chdir(path)
    for file in glob.glob("*.tif"):
        list_id.append(file[:-4])
        # print(file[:-4])
    return list_id

def split_trainval_img_mask(foder_image, foder_image_mask, split):
    foder_name = os.path.basename(foder_image)
    parent = os.path.dirname(foder_image)
    image_list = create_list_id(foder_image)
    np.random.shuffle(image_list)
    count = len(image_list)
    cut_idx = int(round(count*split))
    print(cut_idx)
    train_list = image_list[0:cut_idx]
    # print(train_list)
    # print("_______________")
    # val_list = image_list[cut_idx:count]
    val_list = [id_image for id_image in image_list if id_image not in train_list]
    # print(val_list)

    # tạo  1 folder mới tên là folder_data_split/train/images
    path_train = os.path.join(parent,'Train','images')
    path_train_mask = os.path.join(parent,'Train','masks')
    os.makedirs(path_train, exist_ok=True)
    os.makedirs(path_train_mask, exist_ok=True)

    # tạo 1 folder mới tên là folder_data_split/val/imgges
    path_val = os.path.join(parent,'Test','images')
    path_val_mask = os.path.join(parent,'Test','masks')
    os.makedirs(path_val, exist_ok=True)
    os.makedirs(path_val_mask, exist_ok=True)
        
    # đoạn này chuyển các ảnh sang bên các folder mới
    for image_name in train_list:
        # shutil.copy(os.path.join(foder_image,image_name+'.tif'), path_train)
        os.rename(os.path.join(foder_image,image_name+'.tif'), os.path.join(path_train,image_name+'.tif'))
        os.rename(os.path.join(foder_image_mask,image_name+'.tif'), os.path.join(path_train_mask,image_name+'.tif'))
    for image_name in val_list:
        # shutil.copy(os.path.join(foder_image,image_name+'.tif'), path_val)
        os.rename(os.path.join(foder_image,image_name+'.tif'), os.path.join(path_val,image_name.replace("train","val") +'.tif'))
        os.rename(os.path.join(foder_image_mask,image_name+'.tif'), os.path.join(path_val_mask,image_name.replace("train","val") +'.tif'))
    
    # for fileName in os.listdir(path_val):
    #     os.rename(os.path.join(foder_image,fileName+'.tif'),os.path.join(path_val,fileName.replace("train","val")+'.tif'))

if __name__=='__main__':
    foder_image = os.path.abspath(sys.argv[1])
    foder_image_mask = os.path.abspath(sys.argv[2])
    split = float(sys.argv[3])
    split_trainval_img_mask(foder_image, foder_image_mask, split)

#2 tham so truyen vao:folder anh da _crop, ti le de chia anh
