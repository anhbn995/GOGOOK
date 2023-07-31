import os
from predict_farm import predict_farm
from tqdm import tqdm
import tensorflow as tf
# def create_list_id(path):
    # list_id = []
def create_list_id(path):
    list_id = []
    # dirlist = [os.path.join(path, item) for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    # print(dirlist)
    # for dir_name in dirlist:
    for file in os.listdir(path):
        # if file.endswith(".tif") and (file.startswith("cog_") or file.startswith("COG_")):
        if file.endswith(".tif"):
            list_id.append(os.path.join(path,file))
    return list_id
if __name__=="__main__":
    weight_file = r"./model/farm_indo_load.h5"
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '-i',
        help='Input folder!',
        required=True
    )

    args_parser.add_argument(
        '-o',
        help='Output path',
        required=True
    )

    param = args_parser.parse_args()

    folder_image_path = param.i
    folder_output_path = param.o
    if not os.path.exists(folder_output_path):
        os.makedirs(folder_output_path)
    list_image=create_list_id(folder_image_path)
    size = 480
    model_farm = tf.keras.models.load_model(weight_file)
    for image_path in tqdm(list_image):
        image_name = os.path.basename(image_path)
        outputpredict = os.path.join(folder_output_path,image_name)
        if not os.path.exists(outputpredict):
            print(outputpredict)
            predict_farm(model_farm, image_path, outputpredict, size)
        else:
            pass
    # print(list_image)
