import cv2
import rasterio
import numpy as np

from tqdm import tqdm

def convert_value_obj(big_mask):
    mask_base = big_mask.astype(np.uint8)
    mask_base[mask_base==0]=2
    mask_base[mask_base==1]=0
    mask_base[mask_base==2]=1
    return mask_base

def get_im_by_coord(org_im, start_x, start_y,num_band, padding, crop_size, input_size):
    startx = start_x-padding
    endx = start_x+crop_size+padding
    starty = start_y - padding
    endy = start_y+crop_size+padding
    result=[]
    img = org_im[starty:endy, startx:endx]
    img = img.swapaxes(2,1).swapaxes(1,0)
    for chan_i in range(num_band):
        result.append(cv2.resize(img[chan_i],(input_size, input_size), interpolation = cv2.INTER_CUBIC))
    return np.array(result).swapaxes(0,1).swapaxes(1,2)

def get_img_coords(w, h, padding, crop_size):
    new_w = w + 2*padding
    new_h = h + 2*padding
    cut_w = list(range(padding, new_w - padding, crop_size))
    cut_h = list(range(padding, new_h - padding, crop_size))

    list_hight = []
    list_weight = []
    for i in cut_h:
        if i < new_h - padding - crop_size:
            list_hight.append(i)
    list_hight.append(new_h-crop_size-padding)

    for i in cut_w:
        if i < new_w - crop_size - padding:
            list_weight.append(i)
    list_weight.append(new_w-crop_size-padding)

    img_coords = []
    for i in list_weight:
        for j in list_hight:
            img_coords.append([i, j])
    return img_coords

def padded_for_org_img(values, num_band, padding):
    padded_org_im = []
    for i in range(num_band):
        band = np.pad(values[i], padding, mode='reflect')
        padded_org_im.append(band)

    values = np.array(padded_org_im).swapaxes(0,1).swapaxes(1,2)
    print(values.shape)
    del padded_org_im
    return values

def predict_green(model, values, img_coords, num_band, h, w, padding, crop_size, 
            input_size, batch_size, thresh_hold):
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

def predict_water(model, values, img_coords, num_band, h, w, padding, crop_size, 
            input_size, batch_size, thresh_hold):
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
        y_batch = y_batch[-1]
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

def write_results(mask_base, image_path, result_path):
    with rasterio.open(image_path) as src:
        transform1 = src.transform
        w,h = src.width,src.height
        crs = src.crs
    new_dataset = rasterio.open(result_path, 'w', driver='GTiff',
                                height = h, width = w,
                                count=1, dtype="uint8",
                                crs=crs,
                                transform=transform1,
                                compress='lzw')
    new_dataset.write(mask_base,1)
    new_dataset.close()