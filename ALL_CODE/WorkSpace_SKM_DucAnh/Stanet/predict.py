from data import create_dataset
from models import create_model
from util.util import save_images
import numpy as np
from util.util import mkdir
import argparse
from PIL import Image
import torchvision.transforms as transforms
import glob, os
import torch
import rasterio
from tqdm import tqdm
from rasterio.windows import Window

input_size = 256
crop_size = 200


def write_window_many_chanel(output_ds, arr_c, window_draw_pre):
    s_h, e_h ,s_w, e_w, sw_w, sw_h, size_w_crop, size_h_crop = window_draw_pre 
    output_ds.write(arr_c[s_h:e_h,s_w:e_w],window = Window(sw_w, sw_h, size_w_crop, size_h_crop), indexes = 1)


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        # print('a'*100)
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            # print('b'*100)
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        # print(image_numpy.shape,'t'*100)
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            # print('c'*100)
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # print(image_numpy.shape)
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if normalize:
            # print('d'*100)
            image_numpy = (image_numpy + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    image_numpy = image_numpy.transpose(2,0,1)[0]
    return image_numpy.astype(imtype)


def read_window_and_index_result(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_img_size_model, src_img, num_band_train):
    """
        Trả về img de predict vs kich thước model
        Và vị trí để có thể ghi mask vào trong đúng vị trí ảnh
    """
    if h_crop_start < 0 and w_crop_start < 0:
        # continue
        h_crop_start = 0
        w_crop_start = 0
        size_h_crop = crop_size + padding
        size_w_crop = crop_size + padding
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        tmp_img_size_model[:, padding:, padding:] = img_window_crop
        window_draw_pre = [padding, crop_size + padding, padding, crop_size + padding, start_w_org, start_h_org, crop_size, crop_size]

    # truong hop h = 0 va w != 0
    elif h_crop_start < 0:
        h_crop_start = 0
        size_h_crop = crop_size + padding
        size_w_crop = min(crop_size + 2*padding, w - start_w_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_w_crop == w - start_w_org + padding:
            end_c_index_w =  size_w_crop
            tmp_img_size_model[:,padding:,:end_c_index_w] = img_window_crop
        else:
            end_c_index_w = crop_size + padding
            tmp_img_size_model[:, padding:,:] = img_window_crop
        window_draw_pre = [padding, crop_size + padding ,padding, end_c_index_w, start_w_org, start_h_org,  min(crop_size, w - start_w_org), crop_size]

    # Truong hop w = 0, h!=0 
    elif w_crop_start < 0:
        w_crop_start = 0
        size_w_crop = crop_size + padding
        size_h_crop = min(crop_size + 2*padding, h - start_h_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        if size_h_crop == h - start_h_org + padding:
            end_c_index_h =  size_h_crop
            tmp_img_size_model[:,:end_c_index_h,padding:] = img_window_crop
        else:
            end_c_index_h = crop_size + padding
            tmp_img_size_model[:,:, padding:] = img_window_crop
        window_draw_pre = [padding, end_c_index_h, padding, crop_size + padding, start_w_org, start_h_org, crop_size, min(crop_size, h - start_h_org)]
        
    # Truong hop ca 2 deu khac khong
    else:
        size_w_crop = min(crop_size +2*padding, w - start_w_org + padding)
        size_h_crop = min(crop_size +2*padding, h - start_h_org + padding)
        img_window_crop  = src_img.read([*range(1, num_band_train+1)],window=Window(w_crop_start, h_crop_start, size_w_crop, size_h_crop))
        # print(img_window_crop.shape, size_w_crop, size_h_crop)
        if size_w_crop < (crop_size + 2*padding) and size_h_crop < (crop_size + 2*padding):
            # print(img_window_crop.shape, size_w_crop, size_h_crop)
            end_c_index_h = size_h_crop
            end_c_index_w = size_w_crop
            tmp_img_size_model[:,:end_c_index_h,:   end_c_index_w] = img_window_crop
        elif size_w_crop < (crop_size + 2*padding):
            end_c_index_h = crop_size + padding
            end_c_index_w = size_w_crop
            tmp_img_size_model[:,:,:end_c_index_w] = img_window_crop
        elif size_h_crop < (crop_size + 2*padding):
            end_c_index_w = crop_size + padding
            end_c_index_h = size_h_crop
            tmp_img_size_model[:,:end_c_index_h,:] = img_window_crop
        else:
            end_c_index_w = crop_size + padding
            end_c_index_h = crop_size + padding
            tmp_img_size_model[:,:,:] = img_window_crop
        window_draw_pre = [padding, end_c_index_h, padding, end_c_index_w, start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org)]    
    # print(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w)
    return tmp_img_size_model, window_draw_pre


def transform4():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def predict_win(model, A_img, B_img, numband):
    if numband == 3:
        trans = transform()
    else:
        trans = transform4()
    A = trans(A_img.transpose(1,2,0)).unsqueeze(0)
    B = trans(B_img.transpose(1,2,0)).unsqueeze(0)
    data = {}
    data['A']= A
    data['B'] = B
    # data['A_paths'] = [r"E:\WorkSpaceSkyMap\Change_detection_Dubai\DataTraining\V1\img_unstack\A\E3.tif"]
    model.set_input_predict(data)
    pred = model.test_predict(val=False)
    return tensor2im(pred[0].unsqueeze(0),normalize=False)*255      


def predict_big(opt):

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.eval()

    save_path = opt.results_dir
    os.makedirs(save_path, exist_ok=True)
    print('p'*100, save_path)

    image1_path = opt.image1_path
    image2_path = opt.image2_path
    outputFileName = os.path.join(save_path, os.path.basename(image1_path))
    print(image1_path, '\n', image2_path)

    num_band_train = opt.input_nc
    # num_band_train = 3

    with rasterio.open(image1_path) as src_imgA:
        h,w = src_imgA.height,src_imgA.width
        source_crs = src_imgA.crs
        source_transform = src_imgA.transform
    
    with rasterio.open(outputFileName, 'w', driver='GTiff',
                                    height = h, width = w,
                                    count=1, dtype='uint8',
                                    crs=source_crs,
                                    transform=source_transform,
                                    nodata=0,
                                    compress='lzw') as output_ds:
            output_ds = np.empty((1,h,w))
    
    padding = int((input_size - crop_size)/2)
    list_weight = list(range(0, w, crop_size))
    list_hight = list(range(0, h, crop_size))

    with rasterio.open(outputFileName,"r+") as output_ds:
        with tqdm(total=len(list_hight)*len(list_weight)) as pbar:
            with rasterio.open(image1_path) as src_imgA:
                with rasterio.open(image2_path) as src_imgB:
                    for start_h_org in list_hight:
                        for start_w_org in list_weight:
                            # vi tri bat dau
                            h_crop_start = start_h_org - padding
                            w_crop_start = start_w_org - padding
                            # kich thuoc
                            
                            tmp_A_size_model = np.zeros((num_band_train, input_size,input_size))
                            tmp_A_size_model, window_draw_pre = read_window_and_index_result(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_A_size_model, src_imgA, num_band_train)

                            tmp_B_size_model = np.zeros((num_band_train, input_size,input_size))
                            tmp_B_size_model, window_draw_pre = read_window_and_index_result(h_crop_start, w_crop_start, start_w_org, start_h_org, padding, h, w, tmp_B_size_model, src_imgB, num_band_train)

                            mask_predict_win = predict_win(model, tmp_A_size_model.astype('uint8'), tmp_B_size_model.astype('uint8'), num_band_train)
                            write_window_many_chanel(output_ds, mask_predict_win, window_draw_pre)
                            # img_predict = predict_win(model, tmp_B_size_model, tmp_B_size_model)
                            # print(img_predict.shape)

                            # write_window_many_chanel(output_ds, img_predict, padding, end_c_index_h, padding, end_c_index_w, 
                            #                                 start_w_org, start_h_org, min(crop_size, w - start_w_org), min(crop_size, h - start_h_org))
                            pbar.update()

    

if __name__ == '__main__':
    # 从外界调用方式：
    #  python test.py --image1_path [path-to-img1] --image2_path [path-to-img2] --results_dir [path-to-result_dir]
    import glob
    from tqdm import tqdm
    # list_need_run = glob.glob(r'/home/skm/SKM/WORK/ALL_CODE/WORK/STaNet/Data_test/A/*.tif')
    # for fname in tqdm([os.path.basename(fp) for fp in list_need_run]):
        # print(fname)
        # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # parser.add_argument('--image1_path', type=str, default=f"E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\/tach_ra\/verByte\A\{fname}",
        #                     help='path to images A')
        # parser.add_argument('--image2_path', type=str, default=f"E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\/tach_ra\/verByte\B\{fname}",
        #                     help='path to images B')
        # parser.add_argument('--results_dir', type=str, default=r'E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\tach_ra\verByte\predict', help='saves results here.')

        # parser.add_argument('--name', type=str, default='DUBAI-tonghopQC_ok',
        #                     help='name of the experiment. It decides where to store samples and models')
        # parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        # parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # # model parameters
        # parser.add_argument('--model', type=str, default='CDF0', help='chooses which model to use. [CDF0 | CDFA]')
        # parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB ')
        # parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB')
        # parser.add_argument('--arch', type=str, default='mynet3', help='feature extractor architecture | mynet3')
        # parser.add_argument('--f_c', type=int, default=64, help='feature extractor channel num')
        # parser.add_argument('--n_class', type=int, default=2, help='# of output pred channels: 2 for num of classes')

        # parser.add_argument('--SA_mode', type=str, default='PAM',
        #                     help='choose self attention mode for change detection, | ori |1 | 2 |pyramid, ...')
        # # dataset parameters
        # parser.add_argument('--dataset_mode', type=str, default='changedetection',
        #                     help='chooses how datasets are loaded. [changedetection | json]')
        # parser.add_argument('--val_dataset_mode', type=str, default='changedetection',
        #                     help='chooses how datasets are loaded. [changedetection | json]')
        # parser.add_argument('--split', type=str, default='train',
        #                     help='chooses wihch list-file to open when use listDataset. [train | val | test]')
        # parser.add_argument('--ds', type=int, default='1', help='self attention module downsample rate')
        # parser.add_argument('--angle', type=int, default=0, help='rotate angle')
        # parser.add_argument('--istest', type=bool, default=False, help='True for the case without label')
        # parser.add_argument('--serial_batches', action='store_true',
        #                     help='if true, takes images in order to make batches, otherwise takes them randomly')
        # parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        # parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        # parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        # parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        # parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
        #                     help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        # parser.add_argument('--preprocess', type=str, default='resize_and_crop',
        #                     help='scaling and cropping of images at load time [resize_and_crop | none]')
        # parser.add_argument('--no_flip', type=bool, default=True,
        #                     help='if specified, do not flip(left-right) the images for data augmentation')
        # parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        # parser.add_argument('--epoch', type=str, default='113_F1_1_0.72993',
        #                     help='which epoch to load? set to latest to use latest cached model')
        # parser.add_argument('--load_iter', type=int, default='0',
        #                     help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        # parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        # parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # parser.add_argument('--isTrain', type=bool, default=False, help='is or not')
        # parser.add_argument('--num_test', type=int, default=np.inf, help='how many test images to run')

        # opt = parser.parse_args()
        # predict_big(opt)


    for fp in tqdm(glob.glob(r'/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/A/*.tif')): #[r'/home/skm/SKM16/Tmp/anhtest/resize/image-uint8_RGB/A/A_74.tif']: #
        fname = os.path.basename(fp)
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--image1_path', type=str, default=f"/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/A/{fname}",
                            help='path to images A')
        parser.add_argument('--image2_path', type=str, default=f"/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/B/{fname}",#.replace('T0','T1')
                            help='path to images B')
        parser.add_argument('--results_dir', type=str, default=r'/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/India_256', help='saves results here.')

        """Good"""
        # parser.add_argument('--name', type=str, default='Dubai_change_CDF0_ver256',
        #                     help='name of the experiment. It decides where to store samples and models')
        # parser.add_argument('--epoch', type=str, default='192_F1_1_0.61344',
        #                     help='which epoch to load? set to latest to use latest cached model')

        # parser.add_argument('--name', type=str, default='DUBAI-tonghopQC_ok',
        #                     help='name of the experiment. It decides where to store samples and models')
        # parser.add_argument('--epoch', type=str, default='113_F1_1_0.72993',
        #                     help='which epoch to load? set to latest to use latest cached model')
        
        parser.add_argument('--name', type=str, default='India_256',
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--epoch', type=str, default='42_F1_1_0.48418',
                            help='which epoch to load? set to latest to use latest cached model')
    
        
                            
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        
        # model parameters
        parser.add_argument('--model', type=str, default='CDF0', help='chooses which model to use. [CDF0 | CDFA]')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB ')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB')
        parser.add_argument('--arch', type=str, default='mynet3', help='feature extractor architecture | mynet3')
        parser.add_argument('--f_c', type=int, default=64, help='feature extractor channel num')
        parser.add_argument('--n_class', type=int, default=2, help='# of output pred channels: 2 for num of classes')

        # parser.add_argument('--SA_mode', type=str, default='PAM',
        #                     help='choose self attention mode for change detection, | ori |1 | 2 |pyramid, ...')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='changedetection',
                            help='chooses how datasets are loaded. [changedetection | json]')
        parser.add_argument('--val_dataset_mode', type=str, default='changedetection',
                            help='chooses how datasets are loaded. [changedetection | json]')
        parser.add_argument('--split', type=str, default='train',
                            help='chooses wihch list-file to open when use listDataset. [train | val | test]')
        parser.add_argument('--ds', type=int, default='1', help='self attention module downsample rate')
        parser.add_argument('--angle', type=int, default=0, help='rotate angle')
        parser.add_argument('--istest', type=bool, default=False, help='True for the case without label')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | none]')
        parser.add_argument('--no_flip', type=bool, default=True,
                            help='if specified, do not flip(left-right) the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--isTrain', type=bool, default=False, help='is or not')
        parser.add_argument('--num_test', type=int, default=np.inf, help='how many test images to run')

        opt = parser.parse_args()
        
        predict_big(opt)    

# A3, A11, B2, B10, C8, E3, F3, F11, G3, H7, I1, I7