from osgeo import gdal
def a():
    filename = r"/home/skm/SKM/Tmp_xoa/xoa_luon/T2/S2B_MSIL2A_20210214T031829_N0214_R118_T48PYC_20210214T060910.tif"
    gdal.Open(filename)
img_path = '/home/quyet/DATA_ML/Projects/forest_monitor/Corte_forest/data_train_cloud/cloud_distribute/img'
shp_path = '/home/quyet/DATA_ML/Projects/forest_monitor/Corte_forest/data_train_cloud/cloud_distribute/label'
box_path = '/home/quyet/DATA_ML/Projects/forest_monitor/Corte_forest/data_train_cloud/cloud_distribute/box'


mission = 'green'
use_model = 'unet_3plus'
img_path = '/home/quyet/DATA_ML/Data_work/GreenCover/sentinel2/data_T1_9_sing/green/img'
shp_path = '/home/quyet/DATA_ML/Data_work/GreenCover/sentinel2/data_T1_9_sing/green/label'
box_path = '/home/quyet/DATA_ML/Data_work/GreenCover/sentinel2/data_T1_9_sing/green/box'
old_weights = None