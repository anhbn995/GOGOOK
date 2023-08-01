import os, glob
import numpy as np
import gdalnumeric
import pandas as pd

def cal_area(fp, list_value):
    dict_rs = {}
    for value in list_value:
        raster = gdalnumeric.LoadFile(fp)
        pixel_count = (raster == value).sum()
        area = pixel_count*10*10/10000
        dict_rs.update({value: area})
    return dict_rs



dir_img = r"C:\Users\SkyMap\Desktop\c\7_predict_4class_add_2018_2020"
list_values = [1,2,3,4]
rename_colums = ['1. Rừng', '2. Thảm cỏ', '3. Nước', '4. Khác']


list_fp = glob.glob(os.path.join(dir_img,'*tif'))
rs = {}
for fp in list_fp:
    year = os.path.basename(fp)[:-4]
    rs.update({year: cal_area(fp, list_values)})

df = pd.DataFrame(rs)
print(df)

# doan nay la chuyen bang ve dang cua A Dung can
df['Class'] = ['1. Rừng', '2. Thảm cỏ', '3. Nước', '4. Khác']
df_unpivot = pd.melt(df,id_vars = ['Class'], value_vars=["2018","2019","2020","2021","2022"])
df_unpivot.columns = ['Class', 'Year', 'Area (ha)']
df_unpivot = df_unpivot[['Year', 'Class', 'Area (ha)']]
df_unpivot.to_excel(r'C:\Users\SkyMap\Desktop\c\data\data_area.xlsx', sheet_name='Sheet1', index=False)
print(df_unpivot)



