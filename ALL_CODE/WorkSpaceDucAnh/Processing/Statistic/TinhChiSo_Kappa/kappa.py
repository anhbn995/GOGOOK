import os, glob
import rasterio
import numpy as np 
import pandas as pd
from sklearn.metrics import confusion_matrix


# dir_predict = r"Z:\data_greencover_sing\T1-T9_sing\RUNNN\predict\T2_0.tif"
# dir_label_green = r"Z:\data_greencover_sing\T1-T9_sing\RUNNN\green_label\T2_0.tif"

dir_predict = r"C:\Users\SkyMap\Desktop\Xong_Xoa\T2.tif"
dir_label_green = r"C:\Users\SkyMap\Desktop\Xong_Xoa_mask\T2.tif"

label_green = rasterio.open(dir_label_green).read()[0]
idx_green = np.where(label_green == 255)
label_green[idx_green] = 1
label_green[label_green !=1] = 2


y_true = label_green.flatten() 


predict_mask = rasterio.open(dir_predict).read()[0]
predict_mask[predict_mask != 1] = 2
y_pred = predict_mask.flatten()  


list_name_label = list(range(1,17)) 


cnf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion matrix:')
print(cnf_matrix)
df1 = pd.DataFrame(cnf_matrix, columns = ['green predict','other predict'], index = ['green label', 'other label'])
df1.name = "Unnormalized confusion matrix"


cnf_matrix1 = confusion_matrix(y_true, y_pred, normalize="true")
print('Confusion matrix normal:')
print(cnf_matrix1)
df2 = pd.DataFrame(cnf_matrix1, columns = ['green predict','other predict'], index = ['green label', 'other label'])
df2.name = "Normalizes confusion matrix"


writer = pd.ExcelWriter('KAPPA___oook.xlsx',engine='xlsxwriter')
workbook=writer.book
worksheet=workbook.add_worksheet('Result')
writer.sheets['Result'] = worksheet
worksheet.write_string(0, 0, df1.name)


df1.to_excel(writer,sheet_name='Result',startrow=1 , startcol=0)
worksheet.write_string(df1.shape[0] + 4, 0, df2.name)
df2.to_excel(writer,sheet_name='Result',startrow=df1.shape[0] + 5, startcol=0)
writer.save()
print("oke")