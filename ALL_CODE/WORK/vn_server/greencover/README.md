## GreenCover

Green cover in the world

### Huong dan chay
    Su dung file main.py đầu vào gồm

```` python 
    weight_path_green = "path model green"
    weight_path_water = "path model water"
    weight_path_forest ="path model forest"
    
    fp_in = "path img sentinel 2"
    dir_result_path_green_and_water = 'folder containing temporary images'
    fp_out = "result green cover"
````

Kết quả trung gian nằm ở: *folder containing temporary images* có thể kiểm tra dữ liệu ở đây.

Biến *weight_path_forest* có thể có hoặc "None":
khi không cần chạy forest thì để :```` weight_path_forest = None````

Đường các model được chứa trong ổ mạng: 
 *//192.168.4.4/ml_data/DucAnh/Weight_Good/Model_Greencover*
