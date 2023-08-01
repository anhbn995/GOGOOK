import numpy as np
from PIL import Image
import os, glob

def iou_multi_class(predict, label, num_classes):
    iou_total = 0
    for cls in num_classes:
        predict_cls = np.where(predict == cls, 1, 0)
        label_cls = np.where(label == cls, 1, 0)
        intersection = np.logical_and(predict_cls, label_cls).sum()
        union = np.logical_or(predict_cls, label_cls).sum()
        iou = intersection / union if union != 0 else 0
        iou_total += iou
    mean_iou = iou_total / len(num_classes)
    return mean_iou


def intersection_total_and_union_total(list_fn_img, dir_rs, dir_label, num_classes):
    iou_total = 0
    for cls in num_classes:
        intersection_all_1_class = 0
        union_all_1_class = 0
        for fn_file in list_fn_img:
            predict = np.array(Image.open(os.path.join(dir_rs,fn_file)))
            label = np.array(Image.open(os.path.join(dir_label,fn_file)))
 
            predict_cls = np.where(predict == cls, 1, 0)
            label_cls = np.where(label == cls, 1, 0)
            
            intersection = np.logical_and(predict_cls, label_cls).sum()
            intersection_all_1_class += intersection
            union = np.logical_or(predict_cls, label_cls).sum()
            union_all_1_class += union
        iou = intersection_all_1_class / union_all_1_class if union_all_1_class != 0 else 0
        iou_total += iou
    mean_iou = iou_total / len(num_classes)
    return mean_iou


def main_iou_multi_class_cho_tung_anh(list_fn_img, dir_rs, dir_label, num_classes):
    # num_classes = [1,2,3] gia tri cac lop
    # Load predicted and labeled images
    # dir_rs = r"/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/Union_all_result_cut"
    # for fn_file in [os.path.basename(fp)for fp in glob.glob(os.path.join(dir_rs,'*.tif'))]:
    for idx, fn_file in enumerate(list_fn_img):
        # print(fn_file)
        predict = np.array(Image.open(os.path.join(dir_rs,fn_file)))
        label = np.array(Image.open(os.path.join(dir_label,fn_file)))

        # Compute IOU
        iou = iou_multi_class(predict, label, num_classes)
        print(f'IOU_AOI_{idx+1}:', iou)
     
def main_iou_multi_class_cho_All_Img(list_fn_img, dir_rs, dir_label, num_classes):
    iou = intersection_total_and_union_total(list_fn_img, dir_rs, dir_label, num_classes)
    # iou = iou_multi_class(predict, label, [1,2,3])
    print('IOU_sum_all_AOI:', iou)

if __name__=='__main__':
    list_fn_img = [ "14OCT12031838-S2AS-013302906030_01_P001_cut_0.tif",
                    "14OCT12031838-S2AS-013302906030_01_P001_cut_3.tif",
                    "14JAN21034602-S2AS-013302906010_01_P001_cut_0.tif",
                    "14JAN21034602-S2AS-013302906010_01_P001_cut_1.tif",
                    "14OCT12031838-S2AS-013302906030_01_P001_cut_0.tif"]
    dir_rs_U2net = r"/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/Union_all_result_cut"
    dir_rs_Unet = r"/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/RS_UNET_cut"
    
    dir_label = r'/home/skm/SKM16/A_CAOHOC/ALL_DATA/img_unit8/RS_OKE/Union_all_label'
    num_classes = [1,2,3]
    print('Mode')
    main_iou_multi_class_cho_tung_anh(list_fn_img, dir_rs_U2net, dir_label, num_classes)
    main_iou_multi_class_cho_All_Img(list_fn_img, dir_rs_U2net, dir_label, num_classes)
