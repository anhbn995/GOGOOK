import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from utils.dataloader import predict_dataset
# from CFP_Res2Net import cfpnet_res2net
from collections import OrderedDict
# from pranet import PraNet
from CaraNet import caranet
import rasterio
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
from crop import main_crop_mask
from mosaic import main_mosaic_caranet

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
# parser.add_argument('--pth_path', type=str, default='./snapshots/CaraNet-best_solar/CaraNet-best-solar-88.pth')
parser.add_argument('--pth_path', type=str, default='/home/skm/SKM/WORK/ALL_CODE/WORK/KiemThu/Solar_panel/CaraNet_4band/snapshots/CaraNet-best-solar4band_512/CaraNet-best-solar4band-98.pth')
# data_predict = '/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/izmages_8bit_perizmage/images_per95/tmp_forpredict_big/img_crop512_str256/'
data_predict = '/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/V2/image_8bit_perimage_p98/tmp_forpredict_big/img_crop512_str256/'

opt = parser.parse_args()


now = datetime.now()
time_curent = now.strftime("%Y%b%d-%Hh%Mm%Ss_4band_V2")
path1 = Path(data_predict)
save_path = os.path.join(path1.parent, os.path.basename(opt.pth_path)[:-4] + time_curent)
os.makedirs(save_path, exist_ok=True)
print(save_path)

model = caranet()
weights = torch.load(opt.pth_path)
new_state_dict = OrderedDict()

for k, v in weights.items():
    if 'total_ops' not in k and 'total_params' not in k:
        name = k
        new_state_dict[name] = v
    # print(new_state_dict[k])
        # # print(k)
    # fp = open('./log3.txt','a')
    # fp.write(str(k)+'\n')
    # fp.close()
# print(new_state_dict)
model.load_state_dict(new_state_dict)
model.cuda()
model.eval()

predict_loader = predict_dataset(data_predict, opt.testsize)

list_name = []
for i in tqdm(range(predict_loader.size)):
    image, name = predict_loader.load_data()
    image = image.cuda()

    # res = model(image)
    res5,res4,res2,res1 = model(image)
    res = res5
    res = F.upsample(res, size=(512,512), mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    with rasterio.open(os.path.join(data_predict, name)) as src:
        meta=src.meta
    # meta.update({'count':1,'dtype':'float32'})
    mask = np.array([res])
    mask[mask > 0.3] = 255
    meta.update({'count':1})
    with rasterio.open(os.path.join(save_path,name), 'w', **meta) as dst:
        # dst.write(np.array([res]))
        dst.write(mask)  
    # misc.imsave(save_path+name, res)
    list_name.append(name)


# save_path = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/izmages_8bit_perizmage/images_per95/tmp_forpredict_big/CaraNet-best-solar4band-982022Sep12-08h47m39s_4band"
list_name = [   
            "01_July_Mosaic_P_2",
            "01_July_Mosaic_P_3",
            "01_July_Mosaic_P_4",
            "01_July_Mosaic_P_5",
            "01_July_Mosaic_P_6",
            "02_May_Mosaic_P_2",
            "03_July_Mosaic_P_2"
            ]
outdir_crop_of_mask = main_crop_mask(list_name, save_path)
# outdir_crop_of_mask = r"/home/skm/SKM16/Work/SonalPanel_ThaiLand/1Ver2_lable2/izmages_8bit_perizmage/images_per95/tmp_forpredict_big/CaraNet-best-solar4band-982022Sep12-08h47m39s_4bandcrop_predict"
main_mosaic_caranet(list_name, outdir_crop_of_mask)

