import sys, os
from tqdm import tqdm

sys.path.append(r'E:\WorkSpaceDucAnh')   
from ultils import get_info_txt_with_line_to_list, coppy_file_to_dir


def copy_tif_to_dir_from_txt(fp_txt, dir_contain_image, dir_dest_coppy):
    list_tif_choosen = get_info_txt_with_line_to_list(fp_txt)
    print(list_tif_choosen)
    os.makedirs(dir_dest_coppy, exist_ok=True)

    list_fp_tif_choosen = [os.path.join(dir_contain_image, fname) for fname in list_tif_choosen]
    for fp in tqdm(list_fp_tif_choosen, desc='Copping ...'):
        coppy_file_to_dir(fp, dir_dest_coppy)
    print('Done!')


if __name__=='__main__':
    fp_txt = 'E:\WorkSpaceDucAnh\Tmp\list_file.txt'
    # dir_contain_image = r'E:\WorkSpaceSkyMap\Change_detection_Dubai\Data_Project\img2021_2022'
    dir_contain_image = r'Z:\data_change_detection\stacked\stacked'
    dir_dest_coppy = r'E:\WorkSpaceSkyMap\Change_detection_Dubai\DataTraining\V1\img'
    copy_tif_to_dir_from_txt(fp_txt, dir_contain_image, dir_dest_coppy)

