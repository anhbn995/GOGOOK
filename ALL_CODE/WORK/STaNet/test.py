from data import create_dataset
from models import create_model
from util.util import save_images
import numpy as np
from util.util import mkdir
import argparse
from PIL import Image
import torchvision.transforms as transforms
import glob

def transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def val(opt):
    # image_1_path = opt.image1_path
    # image_2_path = opt.image2_path
    # A_img = Image.open(image_1_path).convert('RGB')
    # B_img = Image.open(image_2_path).convert('RGB')
    # trans = transform()
    # A = trans(A_img).unsqueeze(0)
    # B = trans(B_img).unsqueeze(0)
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    save_path = opt.results_dir
    import os
    os.makedirs(save_path, exist_ok=True)
    print(save_path, 'p'*100)

    model.eval()
    list_img1_fp = glob.glob(os.path.join(opt.image1_path,'*.tif'))
    list_img2_fp = glob.glob(os.path.join(opt.image2_path,'*.tif'))
    for image_1_path, image_2_path in zip(list_img1_fp, list_img2_fp):
        print(os.path.basename(image_1_path))
        A_img = Image.open(image_1_path).convert('RGBA')
        B_img = Image.open(image_2_path).convert('RGBA')
        trans = transform()
        A = trans(A_img).unsqueeze(0)
        B = trans(B_img).unsqueeze(0)
        data = {}
        data['A']= A
        data['B'] = B
        data['A_paths'] = [image_1_path]
        # print(A, B, 'go'*100)
        # print('json'*1000,data)
        model.set_input(data)  # unpack data from data loader
        pred = model.test(val=False)           # run inference return pred

        img_path = [image_1_path]    # get image paths
        save_images(pred, save_path, img_path)



if __name__ == '__main__':
    # 从外界调用方式：
    #  python test.py --image1_path [path-to-img1] --image2_path [path-to-img2] --results_dir [path-to-result_dir]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--image1_path', type=str, default=r'e:\WorkSpaceSkyMap\Change_detection_Dubai\DataTraining\V1\cut256stride128\SplitTrainValTest_addthem_cai_cu\test\A',
                        help='path to images A')
    parser.add_argument('--image2_path', type=str, default=r'e:\WorkSpaceSkyMap\Change_detection_Dubai\DataTraining\V1\cut256stride128\SplitTrainValTest_addthem_cai_cu\test\B',
                        help='path to images B')
    parser.add_argument('--results_dir', type=str, default=r'E:\WorkSpaceTest\STANet\samples\ok112', help='saves results here.')

    parser.add_argument('--name', type=str, default='DUBAI-tonghopQC_ok',
                        help='name of the experiment. It decides where to store samples and models')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    
    # model parameters
    parser.add_argument('--model', type=str, default='CDF0', help='chooses which model to use. [CDF0 | CDFA]')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB ')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB')
    parser.add_argument('--arch', type=str, default='mynet3', help='feature extractor architecture | mynet3')
    parser.add_argument('--f_c', type=int, default=64, help='feature extractor channel num')
    parser.add_argument('--n_class', type=int, default=2, help='# of output pred channels: 2 for num of classes')

    parser.add_argument('--SA_mode', type=str, default='PAM',
                        help='choose self attention mode for change detection, | ori |1 | 2 |pyramid, ...')
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
    parser.add_argument('--epoch', type=str, default='113_F1_1_0.72993',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--load_iter', type=int, default='0',
                        help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
    parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
    parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
    parser.add_argument('--isTrain', type=bool, default=False, help='is or not')
    parser.add_argument('--num_test', type=int, default=np.inf, help='how many test images to run')

    opt = parser.parse_args()
    val(opt)
