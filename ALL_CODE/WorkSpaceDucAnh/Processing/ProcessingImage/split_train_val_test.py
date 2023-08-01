import os, glob, random, shutil
from tqdm import tqdm


def move_list_img_to_outdir(list_fp, out_dir, process_name = 'moving'):
    list_fname_img = []
    for fp in tqdm(list_fp, desc=process_name):
        fp_dest = os.path.join(out_dir, os.path.basename(fp))
        shutil.move(fp, fp_dest)
        list_fname_img.append(os.path.basename(fp))
    return list_fname_img


def move_list_fname_of_list_mask_to_outdir(list_fn, list_mask, out_dir, process_name = 'moving'):
    list_fp_choosen = [fp for fp in list_mask if os.path.basename(fp) in list_fn]
    for fp in tqdm(list_fp_choosen, desc=process_name):
        fp_dest = os.path.join(out_dir, os.path.basename(fp))
        shutil.move(fp, fp_dest)


def split_train_val_test(folder_img, folder_mask, out_dir_split, list_ti_le):
    """
        folder_img: folder chứa ảnh
        folder_mask: folder chứa mask
        list_ti_le: nó chính là tỉ lệ [train, val, test] = [80,10,10], tỉ lệ test có thể là 0 thì nó chỉ chia train và val  
    """
    # check exist test
    check_exist_test = False
    # make dir train/img, /train/mask, val/img, /val/mask, test/img, /test/mask       - nếu có 0 có test thì k tạo folder test.
    dir_train_img = os.path.join(out_dir_split, 'train', 'img')
    os.makedirs(dir_train_img, exist_ok=True)
    dir_train_mask = os.path.join(out_dir_split, 'train', 'mask')
    os.makedirs(dir_train_mask, exist_ok=True)

    dir_val_img = os.path.join(out_dir_split, 'val', 'img')
    os.makedirs(dir_val_img, exist_ok=True)
    dir_val_mask = os.path.join(out_dir_split, 'val', 'mask')
    os.makedirs(dir_val_mask, exist_ok=True)

    if list_ti_le[-1] != 0:
        dir_test_img = os.path.join(out_dir_split, 'test', 'img')
        os.makedirs(dir_test_img, exist_ok=True)
        dir_test_mask = os.path.join(out_dir_split, 'test', 'mask')
        os.makedirs(dir_test_mask, exist_ok=True)
        check_exist_test = True

    # bat dau chia 
    #   - 2 list đường dẫn ảnh và mask
    list_fp_img = glob.glob(os.path.join(folder_img,'*.tif'))
    list_fp_mask = glob.glob(os.path.join(folder_mask,'*.tif'))

    #   - tổng số lượng ảnh
    n_all = len(list_fp_img)

    #   - chuyển phần trăm sang số.
    n_train = round(n_all*list_ti_le[0]/100)
    if check_exist_test:
        n_val = round(n_all*list_ti_le[1]/100)
    else:
        n_val = n_all - n_train

    # bắt đầu lấy ngẫu nhiên 
    # train
    list_fp_train_img = random.sample(list_fp_img, n_train)
    # val
    list_fp_val_all = [x for x in list_fp_img if (x not in list_fp_train_img)]
    list_fp_val_img = random.sample(list_fp_val_all, n_val)

    list_fname_train = move_list_img_to_outdir(list_fp_train_img, dir_train_img, process_name = 'Moving img to train')
    move_list_fname_of_list_mask_to_outdir(list_fname_train, list_fp_mask, dir_train_mask, process_name = 'Moving mask to train')

    print('\n')
    list_fname_val = move_list_img_to_outdir(list_fp_val_img, dir_val_img, process_name = 'Moving img to val')
    move_list_fname_of_list_mask_to_outdir(list_fname_val, list_fp_mask, dir_val_mask, process_name = 'Moving mask to val')

    if check_exist_test:
        print('\n')
        list_fp_test_img= [x for x in list_fp_img if (x not in list_fp_train_img) and (x not in list_fp_val_img)]
        list_fname_test = move_list_img_to_outdir(list_fp_test_img, dir_test_img, process_name = 'Moving img to test')
        move_list_fname_of_list_mask_to_outdir(list_fname_test, list_fp_mask, dir_test_mask, process_name = 'Moving mask to test')

    print('\n Done!')


def create_out_folder(folder_img1, folder_img2, folder_mask, out_dir_split, folder_name):
    name_dir_img1 = os.path.basename(folder_img1)
    name_dir_img2 = os.path.basename(folder_img2)
    name_dir_label = os.path.basename(folder_mask)

    # make dir train/A, /train/B, /train/label, val/A, /val/B, /val/label, test/A, /test/B, /test/label
    out_dir_img1 = os.path.join(out_dir_split, folder_name, name_dir_img1)
    out_dir_img2 = os.path.join(out_dir_split, folder_name, name_dir_img2)
    out_dir_label = os.path.join(out_dir_split, folder_name, name_dir_label)

    os.makedirs(out_dir_img1, exist_ok=True)
    os.makedirs(out_dir_img2, exist_ok=True)
    os.makedirs(out_dir_label, exist_ok=True)
    return out_dir_img1, out_dir_img2, out_dir_label


def split_train_val_test_change_detection(folder_img1, folder_img2, folder_mask, out_dir_split, list_ti_le):
    """
        folder_img1: folder chứa ảnh trước
        folder_img2: folder chứa ảnh sau
        folder_mask: folder chứa mask
        list_ti_le: nó chính là tỉ lệ [train, val, test] = [80,10,10], tỉ lệ test có thể là 0 thì nó chỉ chia train và val  
    """
    # create folder out train, val, test
    dir_train_img1, dir_train_img2, dir_train_label = create_out_folder(folder_img1, folder_img2, folder_mask, out_dir_split, 'train')
    dir_val_img1, dir_val_img2, dir_val_label = create_out_folder(folder_img1, folder_img2, folder_mask, out_dir_split, 'val')
    # check exist test
    if list_ti_le[2] !=0:
        check_exist_test = True
        dir_test_img1, dir_test_img2, dir_test_label = create_out_folder(folder_img1, folder_img2, folder_mask, out_dir_split, 'test')

    # bat dau chia 
    #   - 2 list đường dẫn ảnh và mask
    list_fp_img1 = glob.glob(os.path.join(folder_img1,'*.tif'))
    list_fp_img2 = glob.glob(os.path.join(folder_img2,'*.tif'))
    list_fp_mask = glob.glob(os.path.join(folder_mask,'*.tif'))

    #   - tổng số lượng ảnh
    n_all = len(list_fp_mask)

    #   - chuyển phần trăm sang số và bắt đầu lấy ngẫu nhiên.
    n_train = round(n_all*list_ti_le[0]/100)
    list_fp_train_img1 = random.sample(list_fp_img1, n_train)

    list_fp_val_all1 = [x for x in list_fp_img1 if (x not in list_fp_train_img1)]
    if check_exist_test:
        n_val = round(n_all*list_ti_le[1]/100)
        list_fp_val_img1 = random.sample(list_fp_val_all1, n_val)
        list_fp_test_img1 = [x for x in list_fp_img1 if (x not in list_fp_train_img1) and (x not in list_fp_val_img1)]
    else:
        list_fp_val_img1 = list_fp_val_all1

    list_fname = move_list_img_to_outdir(list_fp_train_img1, dir_train_img1, process_name = 'Moving img1 to train')
    move_list_fname_of_list_mask_to_outdir(list_fname, list_fp_img2, dir_train_img2, process_name = 'Moving img2 to train')
    move_list_fname_of_list_mask_to_outdir(list_fname, list_fp_mask, dir_train_label, process_name = 'Moving label to train')

    print('\n')
    list_fname = move_list_img_to_outdir(list_fp_val_img1, dir_val_img1, process_name = 'Moving img1 to val')
    move_list_fname_of_list_mask_to_outdir(list_fname, list_fp_img2, dir_val_img2, process_name = 'Moving img2 to val')
    move_list_fname_of_list_mask_to_outdir(list_fname, list_fp_mask, dir_val_label, process_name = 'Moving label to val')
    
    if check_exist_test:
        print('\n')
        list_fname = move_list_img_to_outdir(list_fp_test_img1, dir_test_img1, process_name = 'Moving img1 to test')
        move_list_fname_of_list_mask_to_outdir(list_fname, list_fp_img2, dir_test_img2, process_name = 'Moving img2 to test')
        move_list_fname_of_list_mask_to_outdir(list_fname, list_fp_mask, dir_test_label, process_name = 'Moving label to test')

    print('\nDone!')


if __name__ == "__main__":
    """split train val test with bài toán bình thường"""
    # folder_img = r'E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\data_train_model_STANet\cut256\A'
    # folder_mask = r'E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\data_train_model_STANet\cut256\label'
    # out_dir_split = r'E:\WorkSpaceSkyMap\Change_detection_Dubai\Data\data_train_model_STANet\SplitTrainVal'
    # list_ti_le = [80,20,0]
    # split_train_val_test(folder_img, folder_mask, out_dir_split, list_ti_le)


    """split train val test with bài toán change detection với 2 ảnh với 2 folder khác nhau"""
    folder_img1 = r'/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/Img_cut/zgen_cut256stride250/A'
    folder_img2 = r'/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/Img_cut/zgen_cut256stride250/B'
    folder_mask = r'/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/Img_cut/zgen_cut256stride250/label'
    out_dir_split = r'/home/skm/SKM16/Data/WORK/MSRAC/Unstak/uint8/Img_cut/zgen_cut256stride250/SplitTrainValTest'
    list_ti_le = [70,29,1]
    split_train_val_test_change_detection(folder_img1, folder_img2, folder_mask, out_dir_split, list_ti_le)