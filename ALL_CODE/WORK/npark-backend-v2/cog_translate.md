# Parameters:
## Source `source`:
Kiểu dữ liệu: String, Pathlike object hoặc rasterio.io.DatasetReader
Được mở bằng mode "Đọc"

## Destination `dst_path`:
Kiểu dữ liệu: String, Pathlike object
Dataset đầu ra
Được mở bằng mode "Ghi"

## Band indexes `indexes` (optional):
Kiểu dữ liệu: tuple hoặc int
Chỉ số các band muốn copy

## Nodata value `nodata` (optional):
Kiểu dữ liệu: int
Ghi đè giá trị nodata mask của dữ liệu đầu vào

## Data type `dtype` (optional):
Kiểu dữ liệu: str
Thay đổi kiểu dữ liệu cho file đầu ra. Mặc định sẽ trùng với dataset gốc

## Add mask `add_mask` (optional):
Kiểu dữ liệu: bool
Tạo band alpha cho file đầu ra





Input file: 371.8MB
Highest memory occupied: 1.4G



Processing time: 67s (approximate)
