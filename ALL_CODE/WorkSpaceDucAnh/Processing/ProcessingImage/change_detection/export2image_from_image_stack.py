import os, glob
import rasterio
from tqdm import tqdm
def separate_image_stack(fp_stack, number_image_created, out_dir, astype_byte = False):
    if astype_byte:
        print("CHU Y CO CONVERT GIA TRI SANG UNIT8")
    print('Processing ...!')
    with rasterio.open(fp_stack) as src:
        count = src.count
        meta = src.meta

        numband_each_image = count/number_image_created
        if astype_byte:
            meta.update({
                'count': numband_each_image,
                'dtype': 'uint8'
                })
        else:
            meta.update({
                'count': numband_each_image
                })

        if numband_each_image.is_integer():
            name_folder = ord('A')
            for i in range(number_image_created):
                list_band = [*range(int(i*numband_each_image + 1), int((i+1)*numband_each_image+1))]
                dir_out = os.path.join(out_dir, chr(name_folder))
                os.makedirs(dir_out, exist_ok=True)
                fp_out = os.path.join(dir_out, os.path.basename(fp_stack))
                sub_img = src.read(list_band)
                if astype_byte:
                    sub_img = sub_img.astype('uint8')
                with rasterio.open(fp_out, 'w', **meta) as dst:
                    dst.write(sub_img)
                print(f'Done {chr(name_folder)}')
                name_folder += 1
        else:
            print(f'Không thể tách {number_image_created} ảnh từ ảnh {count} băng')



if __name__=='__main__':
    # """ chạy 1 ảnh """
    # fp_stack = r'/home/skm/SKM16/Data/ThaiLandChangeDetection/orthor_ectify_PCI/img_stack.tif'
    # number_image_created = 2
    # out_dir = r'/home/skm/SKM16/Data/ThaiLandChangeDetection/orthor_ectify_PCI/unstack_image'
    # separate_image_stack(fp_stack, number_image_created, out_dir)

    """chạy cả folder"""
    dir_img_stack = r"/home/skm/SKM16/Planet_GreenChange/Data_4band_Green/ImgRGBNir"
    out_dir = r"/home/skm/SKM16/Planet_GreenChange/Data_4band_Green/ImgRGBNir_unstack"
    number_image_created = 2
    list_fp_img_stack = glob.glob(os.path.join(dir_img_stack, '*.tif'))
    for fp_stack in tqdm(list_fp_img_stack, desc=f'Tách ảnh {number_image_created}'):
        separate_image_stack(fp_stack, number_image_created, out_dir)
