from osgeo import gdal, osr

def align_raster(input_raster, output_raster, reference_raster):
    # Mở tệp tin raster đầu vào
    input_ds = gdal.Open(input_raster, gdal.GA_ReadOnly)
    input_proj = input_ds.GetProjection()
    input_geotrans = input_ds.GetGeoTransform()

    # Mở tệp tin raster tham chiếu
    reference_ds = gdal.Open(reference_raster, gdal.GA_ReadOnly)
    reference_proj = reference_ds.GetProjection()
    reference_geotrans = reference_ds.GetGeoTransform()
    reference_width = reference_ds.RasterXSize
    reference_height = reference_ds.RasterYSize

    # Tạo tệp tin đầu ra với thông số từ tệp tin tham chiếu
    output_ds = gdal.GetDriverByName('GTiff').Create(output_raster, reference_width, reference_height, 1, gdal.GDT_Float32)
    output_ds.SetProjection(reference_proj)
    output_ds.SetGeoTransform(reference_geotrans)

    # Căn chỉnh raster
    gdal.ReprojectImage(input_ds, output_ds, input_proj, reference_proj, gdal.GRA_Bilinear)

    # Đóng tệp tin
    input_ds = None
    reference_ds = None
    output_ds = None

# Sử dụng hàm để căn chỉnh raster
input_raster = r"E:\Slope_monitoring\result_final2\S1A_IW_GRDH_1SDV_20220323T215024_0.tif"
output_raster = r"E:\Slope_monitoring\result_final2\S1A_IW_GRDH_1SDV_20220323T215024_0_output.tif"
reference_raster = r"E:\Slope_monitoring\result_final2\S1A_IW_GRDH_1SDV_20220416T215025_0.tif"

align_raster(input_raster, output_raster, reference_raster)
