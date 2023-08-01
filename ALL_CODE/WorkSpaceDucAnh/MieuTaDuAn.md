# Dự án nhận diện sự thay đổi ở Dubai, Barain với model Stanet.

## Bước thực hiện
Bước 1: 
- Registation các ảnh lại với nhau.
Bước 2:
- Stack chúng lại với nhau, bằng code stack.
Bước 3:
- Sinh mask bằng code sinh mask:
```bash
    ProcessingImage\build_mask.py
```
Bước 4:
- Chia ảnh đã stack ra thành các ảnh riêng biệt theo thư mục A,B,C ... tùy theo ảnh stack là ghep của mấy ảnh.
```bash
    ProcessingImage\change_detection\export2image_from_image_stack.py
```
Bước 5:
- Cắt nhỏ các cặp ảnh trước, sau, mask thành những ảnh nhỏ với kích thước mẫu 256*256
```bash 
    ProcessingImage\crop_and_stride_image.py
```
Bước 6:
- Xóa bớt lượng ảnh nodata ra khỏi dữ liệu luyện
```bash
    ProcessingImage\remove_nodata_with_percent.py
```
Bước 7:
- Chia dữ liệu vừa cắt ra thành các tập train, val, test
```bash
    ProcessingImage\split_train_val_test.py
```
Bước 8:
- Đưa dữ liệu vào luyện.