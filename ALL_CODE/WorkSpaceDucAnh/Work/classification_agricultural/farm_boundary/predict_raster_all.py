import os
from predict_farm import predict_farm
from tqdm import tqdm
import tensorflow as tf
def create_list_id(path):
    list_id = []
    
    dirlist = [os.path.join(path, item) for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    print(dirlist)
    for dir_name in dirlist:
        for file in os.listdir(dir_name):
            if file.endswith(".tif") and file.startswith("COG"):
                list_id.append(os.path.join(dir_name,file))
    return list_id
if __name__=="__main__":
    folder_image_path = r"G:\EOF_mosaic\FINAL_MOSAIC_GXL"
    weight_file = r"G:\EOF_farm_boundary_v2\model\model_farm.h5"
    folder_output_path = r"G:\EOF_farm_boundary_v2\result\ressult_farm_0406"
    list_image=create_list_id(folder_image_path)
    size = 480
    model_farm = tf.keras.models.load_model(weight_file)
    for image_path in tqdm(list_image[::-1]):
        image_name = os.path.basename(image_path)
        outputpredict = os.path.join(folder_output_path,image_name)
        if not os.path.exists(outputpredict):
            print(outputpredict)
            predict_farm(model_farm, image_path, outputpredict, size)
        else:
            pass
    # print(list_image)