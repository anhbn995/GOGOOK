from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm
import threading
import glob, os
import cv2
import tensorflow as tf
# import Vectorization

# from rio_tiler.io import COGReader
# from tensorflow.compat.v1.keras.backend import set_session
import os, glob
import json


"""
    
    step 1: u2net : get tọa do ô chứa msssv, điểm 
        - input : list d dan anh
        - input_2 : thu muc out json
        - output : json : tuong ung vs moi anh

"""

# warnings.filterwarnings("ignore")
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.compat.v1.Session(config=config))
num_bands = 3
size = 256
def predict_edge(model, path_image, path_predict, size=256):

    
    img = Image.open(path_image)
    width, height = img.size
    print(img.size)
    input_size = size
    stride_size = input_size - input_size // 4
    padding = int((input_size - stride_size) / 2)

    list_coordinates = []
    for start_y in range(0, height, stride_size):
        for start_x in range(0, width, stride_size):
            x_off = start_x if start_x == 0 else start_x - padding
            y_off = start_y if start_y == 0 else start_y - padding

            end_x = min(start_x + stride_size + padding, width)
            end_y = min(start_y + stride_size + padding, height)

            x_count = end_x - x_off
            y_count = end_y - y_off
            list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))

    num_bands = 3 
    image_data = np.array(img)

    with Image.new('L', (width, height)) as result_img:
        result_data = np.array(result_img)

        read_lock = threading.Lock()
        write_lock = threading.Lock()

        def process(coordinates):
            x_off, y_off, x_count, y_count, start_x, start_y = coordinates
            read_wd = (x_off, y_off, x_off + x_count, y_off + y_count)
            with read_lock:
                values = image_data[y_off:y_off + y_count, x_off:x_off + x_count, :num_bands]

            if image_data.dtype == 'uint8':
                image_detect = values.astype(int)
        

            img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
            mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding), (padding, padding)))
            shape = (stride_size, stride_size)

            if y_count < input_size or x_count < input_size:
                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                mask = np.zeros((input_size, input_size), dtype=np.uint8)

                if start_x == 0 and start_y == 0:
                    img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                    mask[(input_size - y_count):input_size - padding, (input_size - x_count):input_size - padding] = 1
                    shape = (y_count - padding, x_count - padding)
                elif start_x == 0:
                    img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                    if y_count == input_size:
                        mask[padding:y_count - padding, (input_size - x_count):input_size - padding] = 1
                        shape = (y_count - 2 * padding, x_count - padding)
                    else:
                        mask[padding:y_count, (input_size - x_count):input_size - padding] = 1
                        shape = (y_count - padding, x_count - padding)
                elif start_y == 0:
                    img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                    if x_count == input_size:
                        mask[(input_size - y_count):input_size - padding, padding:x_count - padding] = 1
                        shape = (y_count - padding, x_count - 2 * padding)
                    else:
                        mask[(input_size - y_count):input_size - padding, padding:x_count] = 1
                        shape = (y_count - padding, x_count - padding)
                else:
                    img_temp[0:y_count, 0:x_count] = image_detect
                    mask[padding:y_count, padding:x_count] = 1
                    shape = (y_count - padding, x_count - padding)

                image_detect = img_temp

            mask = (mask != 0)

            if np.count_nonzero(image_detect) > 0:
                if len(np.unique(image_detect)) <= 2:
                    pass
                else:
                    y_pred = model.predict(image_detect[np.newaxis, ...] / 255.)

                    # Chuyển đổi thành numpy array
                    y_pred = np.array(y_pred)

                    # Áp dreeshold
                    y_pred = (y_pred[0, 0, ..., 0] > 0.2).astype(np.uint8)

                    y = y_pred[mask].reshape(shape)
         

                    with write_lock:
                        result_data[start_y:start_y + shape[0], start_x:start_x + shape[1]] = y

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

        result_img.putdata(result_data.flatten())
        result_img.save(path_predict)

        return path_predict , path_image

def sort_list_to_row(cell):
    thresh = sorted(cell, key=lambda x: [x[1],x[0]])
    # thresh = sorted(cell, key=lambda x: x[0])
    epsilon = 45
    min_x = thresh[0][1]
    list_row = []
    list_tmp = []
    for i in range(len(thresh)):
        box_x = thresh[i]
        if abs(box_x[1] - min_x) < epsilon:
            list_tmp.append(box_x)
        else:
            list_row.append(list_tmp)
            list_tmp = []
            list_tmp.append(box_x)
            min_x = box_x[1]
        if i ==len(thresh)-1:
            list_row.append(list_tmp)
    return list_row
def sort_list_to_col(list_row):
    list_result = []
    for row in list_row:
        thresh = sorted(row, key=lambda x: x[0])
        list_result.append(thresh)

    return list_result





def get_coordinates_all(input_path,input_mask):
    image = Image.open(input_path)
    image = np.array(image)
    
    # print(image.shape)
    re = np.zeros_like(image)
    re = image.astype(np.uint8)
    bgr_image = cv2.convertScaleAbs(re)

    mask = np.array(Image.open(input_mask))
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # print(mask.shape)
    w,h = mask.shape[0], mask.shape[1]

    masked_image = cv2.bitwise_and( image,  image, mask=mask)
    masked_image = masked_image.astype(np.uint8)
    gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(len(contours))

    cell = []
    toa_do_x = []
    width_ = []
    for contour in contours:

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            if h<30 or h > 300:
                continue
    
        # cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            else:

                cell.append([x,y,w,h])
                width_.append(w)
                toa_do_x.append(x)
    for box in cell:
        x,y,w,h = box
        cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    # import matplotlib.pyplot as plt
    # imgplot = plt.imshow(bgr_image)
    # plt.show()
    list_sort_raw = sort_list_to_row(cell)
    list_sort_col = sort_list_to_col(list_sort_raw)
    return list_sort_col,input_path

def get_infomation(list_sort_col,col_select,input_path,dir_out_json):
    #col select [0:6]

    col_mssv = col_select[1]
    col_diem = col_select[2]
    col_stt = col_select[0]
    ID = 1
    dict_box = {
        ID :{

            'MSSV' :[],
            'DIEM': [],
            'STT' : [],
        }

        }
    for i in range(len(list_sort_col)):
        # print(i)
        # print(list_sort_col[i])
        # print(len(list_sort_col[i]) )
        if i == 0:
            continue
        if len(list_sort_col[i])-1 < col_diem:
            ID = i

            if ID in dict_box:
                dict_box[ID]['MSSV'].append('NULL')
                dict_box[ID]['DIEM'].append('NULL')
                dict_box[ID]['STT'].append('NULL')
            else:
                # Khởi tạo ID trong dict_box nếu chưa tồn tại
                dict_box[ID] = {
                    'MSSV': ['NULL'],
                    'DIEM': ['NULL'],
                    'STT': ['NULL']
                }
          
        
        
        else:
            ID = i

            if ID in dict_box:
                dict_box[ID]['MSSV'].append(list_sort_col[i][col_mssv])
                dict_box[ID]['DIEM'].append(list_sort_col[i][col_diem])
                dict_box[ID]['STT'].append(list_sort_col[i][col_stt])
            else:
                # Khởi tạo ID trong dict_box nếu chưa tồn tại
                dict_box[ID] = {
                    'MSSV': [list_sort_col[i][col_mssv]],
                    'DIEM': [list_sort_col[i][col_diem]],
                    'STT': [list_sort_col[i][col_stt]]


    }
    
    print(dict_box)
    outjson = os.path.join(dir_out_json,os.path.basename(input_path).replace('.PNG','.json'))
    

    with open(outjson, "w") as json_file:
        json.dump(dict_box, json_file)
    return outjson
    

    # a = list_sort_col[-1]
    # for i_tmp in a:
    #     x,y,w,h = i_tmp
    # thresh = sorted(cell, key=lambda x: [x[1],x[0]])
    # print(len(cell))
    # for box in thresh[0:35]:
    #     x,y,w,h = box
    #     cv2.rectangle(bgr_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
    # import matplotlib.pyplot as plt
    # imgplot = plt.imshow(bgr_image)
    # plt.show()
    # return dict_box
    print(list_sort_col[2])
    # id_diem=0

    # # dict_box = {
    # #     'MSSV' :{},
    # #     'DIEM': {}
    # # }
    # dict_box_all = {
    #     'BOX_ALL':{}
    # }
    # #x,y,w,h

    # # id_mssv = 0
    # # mssv = []
    # # diem = []
    # # for box_x in thresh:

    # #     for box_y in thresh:
           
    # #         if box_x[0]  and min_mssv < box_x[2] < max_mssv :
    # #             # cv2.rectangle(bgr_image, (box_x[0], box_x[1]), (box_x[0] + box_x[2], box_x[1] + box_x[3]), (0, 255, 255), 2)
    # #             cv2.rectangle(bgr_image, (box_x[0], box_x[1]), (box_x[0] + box_x[2], box_x[1] + box_x[3]), (0, 255, 255), 2)
    # # # import matplotlib.pyplot as plt
    # # # # import matplotlib.image as mpimg

    # # # imgplot = plt.imshow(bgr_image)
    # # # plt.show()
    # # # <5
    # #             if box_x[0] != box_y[0] and (abs(box_x[1] - box_y[1])<10) and min_diem < box_y[2] < max_diem:
    # #                 cv2.rectangle(bgr_image, (box_y[0], box_y[1]), (box_y[0] + box_y[2], box_y[1] + box_y[3]), (0, 0, 255), 2)
    # #                 mssv_x ,mssv_y,mssv_w,mssv_h =   box_x[0], box_x[1],box_x[2],box_x[3]
    # #                 diem_x, diem_y, diem_w , diem_h =  box_y[0], box_y[1],box_y[2],box_y[3]
    # #                 mssv.append([mssv_x ,mssv_y,mssv_w,mssv_h])
    # #                 diem.append([diem_x, diem_y, diem_w , diem_h ])
    # #                 # print(mssv)
    # #                 # print(diem)
    # #                 id_mssv+=1
    # #                 id_diem+=1
    # #                 dict_box['MSSV'][id_mssv] = mssv
    # #                 dict_box['DIEM'][id_diem] = diem

    # #                 mssv = []
    # #                 diem = []
    # outjson = os.path.join(dir_out_json,os.path.basename(input_path).replace('.PNG','.json'))
    

    # with open(outjson, "w") as json_file:
    #     json.dump(dict_box, json_file)
    # return outjson
    # import matplotlib.pyplot as plt
    # # import matplotlib.image as mpimg

    # imgplot = plt.imshow(bgr_image)
    # plt.show()
    # return dict_box




if __name__=="__main__":
    model_path = r'/home/skymap/data/FLASK_API/module/weights/u2net_256_transcript_mrsac_v1_model.h5'
   
    dir_img = '/home/skymap/data/CHUYENDOISOVT/ada/out/imageLSFAN_0.PNG'
    dir_out = '/home/skymap/data/CHUYENDOISOVT/ada/out/1'

    # dir_out_json = '/home/skymap/data/CHUYENDOISOVT/FINAL_PROCESS/STEP2/'
    
    model_detect = tf.keras.models.load_model(model_path)

    output_path = os.path.join(dir_out,os.path.basename(dir_img))
    path_mask,path_image = predict_edge(model_detect, dir_img, output_path, size)
    # path_image = '/home/skymap/data/CHUYENDOISOVT/OUTDATA/create/4/STEP1/CROP/27.PNG'
    # path_mask = '/home/skymap/data/CHUYENDOISOVT/OUTDATA/create/4/STEP2/edge/27.PNG'
    # list_to_col,input_path = get_coordinates_all(path_image,path_mask)
    # # print(list_to_col)
    # col_select = [0,1,4]
    # out_json = get_infomation(list_to_col,col_select,input_path,dir_out_json)
   
  
            
                

