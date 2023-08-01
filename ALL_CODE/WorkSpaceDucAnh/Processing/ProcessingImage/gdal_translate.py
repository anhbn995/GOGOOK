from osgeo import gdal
import os
import glob
from tqdm import tqdm

def tran(input_file, output_file):
    # output_folder
    # Đường dẫn đến tệp tin nguồn
    # input_file = '/path/to/input/file'

    # # Đường dẫn đến tệp tin đích
    # output_file = '/path/to/output/file'

    # Định dạng đầu ra (ví dụ: GeoTIFF)
    output_format = 'GTiff'

    # Tạo đối tượng GDALDataset từ tệp tin nguồn
    input_dataset = gdal.Open(input_file)

    # Kiểm tra xem tệp tin nguồn có hợp lệ hay không
    if input_dataset is not None:
        # Lấy số băng của ảnh nguồn
        num_bands = input_dataset.RasterCount

        # Tạo đối tượng GDALDataset cho tệp tin đầu ra
        output_dataset = gdal.GetDriverByName('GTiff').Create(output_file, input_dataset.RasterXSize,
                                                            input_dataset.RasterYSize, num_bands, gdal.GDT_UInt16)

        # Sao chép dữ liệu từ từng băng nguồn sang từng băng đích
        for band_num in range(1, num_bands + 1):
            output_dataset.GetRasterBand(band_num).WriteArray(input_dataset.GetRasterBand(band_num).ReadAsArray())

            # Ghi thông tin về hệ thống tọa độ và biến đổi từ ảnh nguồn sang ảnh đích
            output_dataset.SetProjection(input_dataset.GetProjection())
            output_dataset.SetGeoTransform(input_dataset.GetGeoTransform())

        # Đóng đối tượng dataset đầu ra
        output_dataset = None

        # Đóng đối tượng dataset nguồn
        input_dataset = None
    else:
        print("Failed to open the input file.")

if __name__=="__main__":
    list_name_dir = ["AOI_1","AOI_2","AOI_3","AOI_7","AOI_8","AOI_9","AOI_10","AOI_11","AOI_12","AOI_14","AOI_15"]
    dir_img = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Regis_4band_original/Image_origin"
    dir_cog_out = r"/home/skm/SKM16/Planet_GreenChange/2_Indonesia_Mining_Exhibition_Data/Regis_4band_original/Image_origin_cog"
    
    for fname in list_name_dir:
        dir_img_AOI = os.path.join(dir_img, fname)
        dir_cog_out_AOI = os.path.join(dir_cog_out, fname)
        os.makedirs(dir_cog_out_AOI, exist_ok=True)
        
        for fp_img in glob.glob(os.path.join(dir_img_AOI, '*.tif')):
        ## Sửa chính ảnh đó
            # fp_img = r"/home/skm/SKM16/Data/ThaiLandChangeDetection/Building_change_stanet/image8band_unstack_rgb (copy)/B/stack.tif"
            # _translate(fp_img, fp_img)
            

        # # tạo file khác thì
            # fp_img_rs = fp_img.replace('.tif','_cog.tif')
            fp_img_rs = os.path.join(dir_cog_out_AOI, os.path.basename(fp_img))
            tran(fp_img, fp_img_rs)