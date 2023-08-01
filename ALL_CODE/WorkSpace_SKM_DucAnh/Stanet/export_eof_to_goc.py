import os
import uuid
import shutil
import pandas as pd
from tqdm import tqdm

df = pd.read_csv('/home/skm/SKM/WORK/ALL_CODE/WORK/STaNet/export_eof.cvs')
T0 = df[df['folder_id'] == 10117]
T1 = df[df['folder_id'] == 10118]
frames = [T0, T1]
df_all = result = pd.concat(frames)
# print(df_all)

fp_img_all = r'/home/skm/eodata/tmp_hieu/d25b373cc8984c698b43d1184d713706'
fp_dir_out = r'/home/skm/SKM16/Data/WORK/Bahrain_Change/Data_Dao'
os.makedirs(fp_dir_out, exist_ok=True)

with tqdm(total=len(df_all)) as pbar: 
    for i in tqdm(df_all.iloc):
        idx_img = uuid.UUID(i["id"]).hex + '.tif'
        name_folder = i["folder_id"]
        name_file = i["name"] + '.tif'
        
        if name_folder == 10117:
            dir_name = 'T0'
        else:
            dir_name = 'T1'
            
        fp_src = os.path.join(fp_img_all, idx_img)
        fp_dst = os.path.join(fp_dir_out, dir_name, name_file)
        shutil.copy2(fp_src, fp_dst)
        # break
        pbar.update(1)