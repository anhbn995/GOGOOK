import cv2
import numpy as np

from tqdm import tqdm
from osgeo import gdal
from download_and_processing_cloud.greencover_api.forest.convert_tif import dilation_obj, remove_small_items, write_image
from download_and_processing_cloud.greencover_api.green_cover.utils import get_img_coords, padded_for_org_img, get_im_by_coord


def predict(model, values, img_coords, num_band, h, w, padding, crop_size, 
            input_size, batch_size, thresh_hold, choose_stage):
    cut_imgs = []
    for i in range(len(img_coords)):
        im = get_im_by_coord(values, img_coords[i][0], img_coords[i][1],
                            num_band,padding, crop_size, input_size)
        cut_imgs.append(im)

    a = list(range(0, len(cut_imgs), batch_size))

    if a[len(a)-1] != len(cut_imgs):
        a[len(a)-1] = len(cut_imgs)

    y_pred = []
    for i in tqdm(range(len(a)-1)):
        x_batch = []
        x_batch = np.array(cut_imgs[a[i]:a[i+1]])
        y_batch = model.predict(x_batch)
        if len(model.outputs)>1:
            y_batch = y_batch[choose_stage]
        y_pred.extend(y_batch)
    big_mask = np.zeros((h, w)).astype(np.float16)
    for i in range(len(cut_imgs)):
        true_mask = y_pred[i].reshape((input_size,input_size))
        true_mask = (true_mask>thresh_hold).astype(np.uint8)
        true_mask = (cv2.resize(true_mask,(input_size, input_size), interpolation = cv2.INTER_CUBIC)>thresh_hold).astype(np.uint8)
        start_x = img_coords[i][1]
        start_y = img_coords[i][0]
        big_mask[start_x-padding:start_x-padding+crop_size, start_y-padding:start_y -
                    padding+crop_size] = true_mask[padding:padding+crop_size, padding:padding+crop_size]
    del cut_imgs
    return big_mask

def inference(model, weight_path, image_path, result_path, img_size=128, crop_size=100, num_band=4,
            batch_size=2, thresh_hold=0.5, dil=False, rm_small= False, choose_stage=None, return_img=True):
    infer_model = model.load_weights(weight_path) 

    dataset = gdal.Open(image_path)
    values = dataset.ReadAsArray()[0:num_band]/255
    h,w = values.shape[1:3]    
    padding = int((img_size - crop_size)/2)
    img_coords = get_img_coords(w, h, padding, crop_size)
    values = padded_for_org_img(values, num_band, padding)
    big_mask = predict(model, values, img_coords, num_band, h, w, padding, crop_size, 
                        img_size, batch_size, thresh_hold, choose_stage)
    
    if dil:
        big_mask = dilation_obj(big_mask)
    
    if rm_small:
        big_mask = remove_small_items(big_mask, threshhlod_rm_holes=256,
                                        threshhold_rm_obj=100)
    
    big_mask[big_mask==0]=2
    big_mask[big_mask==1]=0
    big_mask[big_mask==2]=1
    if return_img:
        result_path = write_image(image_path, result_path, big_mask)
        return result_path
    else:
        return big_mask