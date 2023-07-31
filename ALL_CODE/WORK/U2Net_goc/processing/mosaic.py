import os
import sys
import shutil
from glob import glob
from unicodedata import name
from skyamqp import AMQP_Client

def _prepare_input_file(temporary_folder, file_list):
    input_text = f'{temporary_folder}/input.txt'
    print(input_text)
    input_content = ''

    for file_path in file_list:
        input_content += f'\"{file_path}\"\n'

    with open(input_text, 'w') as cursor:
        cursor.write(input_content)
    return input_text


def connect_to_client():
    connection = AMQP_Client(host='192.168.4.100',
                            port='5672',
                            virtual_host='/eof',
                            username='eof_rq_worker',
                            password='123',
                            heartbeat=5)
    print('**Connected AMQP host!!')
    return connection

def move_img(img_folder, out_path):
    list_img = glob(os.path.join(img_folder,'*.tif'))
    # print(img_folder)
    print("**List image mosaic:", list_img)
    for i in list_img:
        name_img = os.path.basename(i)
        in_folder = i
        out_folder = os.path.join(out_path, name_img)
        shutil.copyfile(in_folder, out_folder)

def check_base_img(base_img):
    if os.path.exists(base_img):
        try:
            img_base = glob(os.path.join(base_img,'*.tif'))[0]
        except:
            return False
        return img_base
    else:
        return False

def client_run_pci(input_folder, cloud_tmp_dir, temp_folder, base_img, name):
    if os.path.exists(cloud_tmp_dir):
        pass
    else:
        os.mkdir(cloud_tmp_dir)

    if os.path.exists(temp_folder):
        pass
    else:
        os.mkdir(temp_folder)
    
    if base_img:
        print("**Merge with base image**")
        cloud_tmp_dir_base = os.path.join(cloud_tmp_dir, 'AAA_base.tif')
        shutil.copyfile(base_img, cloud_tmp_dir_base)
    else:
        print("**Merge without base image**")

    move_img(input_folder, cloud_tmp_dir)
    list_file_path_mosaic = []
    for path in glob(os.path.join(cloud_tmp_dir,'*.tif')):
        list_file_path_mosaic.append(path)
    list_file_path_mosaic = sorted(list_file_path_mosaic)
    input_file = _prepare_input_file(temp_folder, list_file_path_mosaic)
    pre_translate_path = f'{temp_folder}/%s.tif'%(name)

    connection = connect_to_client()
    gxl_rpcClient = connection.create_RPC_Client('gxl-python')
    print(input_file)
    print(pre_translate_path)
    response = gxl_rpcClient.send('mosaic', {
        "mfile": input_file,
        "out_path": pre_translate_path
    })
    print(response)
    if not response['success']:
        raise Exception(response['message'])
    print("**Finished merge and mosaic")
    return pre_translate_path

def main_mosaic_pci(results_img, out_img_cloud, base_path, cloud_tmp_dir, temp_folder, name, run_agian=True):
    if not os.path.exists(cloud_tmp_dir):
        os.mkdir(cloud_tmp_dir)
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    cloud_tmp_geotest= os.path.join(cloud_tmp_dir, name)
    if not os.path.exists(results_img) or run_agian:
        print("Merge image and mosaic with PCI...")
        out_merge_mosaic = client_run_pci(out_img_cloud, cloud_tmp_geotest, temp_folder, base_path, name)
        shutil.copyfile(out_merge_mosaic, results_img)
        print("\n")


if __name__=="__main__":
    name = 'xxx'
    results_month_path = '/mnt/Nam/public/Linh/'
    results_img = os.path.join(results_month_path, name + '_mosaic_pci.tif')
    out_img_cloud = '/mnt/Nam/public/Linh/mosaic_mongolia'
    if not os.path.exists(out_img_cloud):
        os.mkdir(out_img_cloud)

    # if os.path.exists(results_img):
    #     os.replace(results_img, results_img.replace(results_month_path, out_img_cloud))

    base_path = None
    cloud_tmp_dir = '/mnt/Nam/geoai_data_test/mosaic'
    temp_folder = '/mnt/Nam/geoai_data_test/temp/mosaic_pci'
    main_mosaic_pci(results_img, out_img_cloud, base_path, cloud_tmp_dir, temp_folder, name)
    os.remove(out_img_cloud)
    os.remove(temp_folder)
    os.remove(cloud_tmp_dir)
    sys.exit()