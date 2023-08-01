import os, glob
import pandas as pd

dir_fp_csv = r"C:\Users\SkyMap\Desktop\b\7_predict_4class"
list_csv = glob.glob(os.path.join(dir_fp_csv, '*csv'))

frames = []
for fp_csv in list_csv:
    year =  os.path.basename(fp_csv)[:-4]
    df = pd.read_csv(fp_csv)
    df.columns = ['Xã', '1. Rừng', '2. Thảm cỏ', '3. Nước', '4. Khác', 'Đơn vị']
    df['Year'] = year
    new_cols = ['Year', 'Xã', '1. Rừng', '2. Thảm cỏ', '3. Nước', '4. Khác', 'Đơn vị']
    df = df.reindex(columns=new_cols)
    frames.append(df)

df_result = pd.concat(frames)
# df_result.to_csv(r'C:\Users\SkyMap\Desktop\b\cap_xa_dang_cot.csv')
df_result.to_excel(r'C:\Users\SkyMap\Desktop\c\data_cap_xa\cap_xa_colums.xlsx', sheet_name='Sheet1', index=False)
print(df_result)


df_unpivot = pd.melt(df_result, id_vars=['Year','Xã'], value_vars=['1. Rừng', '2. Thảm cỏ', '3. Nước', '4. Khác'])
df_unpivot.rename(columns = {'Xã':'Xa', 'variable':"Class", 'value':'Area (ha)'}, inplace = True)
# df_unpivot.to_csv(r'C:\Users\SkyMap\Desktop\b\cap_xa_dang_hang.csv')
df_unpivot.to_excel(r'C:\Users\SkyMap\Desktop\c\data_cap_xa\cap_xa_row.xlsx', sheet_name='Sheet1', index=False)
print(df_unpivot)