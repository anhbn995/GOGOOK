import tensorflow as tf
import numpy as np
import geopandas as gpd
import glob
import os
import uuid
np.random.seed(1000)


def get_list_name_file(path_dir, name_file = '*.tif'):
    list_file_dir = []
    for file_ in glob.glob(os.path.join(path_dir, name_file)):
        # head, tail = os.path.split(file_)
        list_file_dir.append(file_)
    return list_file_dir


def append_df_by_shp(list_fp):
    df = gpd.read_file(list_fp[0])
    for fp in list_fp[1:]:
        df_ = gpd.read_file(fp)
        df = df.append(df_)
    return df


def nms_shp(dir_shp, out_file):
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    list_path_file = get_list_name_file(dir_shp, '*.shp')
    df = append_df_by_shp(list_path_file)
    # crs_origin = df2.crs
    dir_tmp = os.path.join(dir_shp, uuid.uuid4().hex)
    os.makedirs(dir_tmp, exist_ok=True)
    tmp_file = os.path.join(dir_tmp,'tmp.shp')
    df.to_file(tmp_file)
    df = gpd.read_file(tmp_file)
    df_bound = df.bounds
    print(len(df_bound))
    df_bound = df_bound.to_numpy()
    # score = np.random.rand(df_bound.shape[0]).tolist()
    score = np.ones(df_bound.shape[0]).tolist()
    result = tf.image.non_max_suppression(df_bound.tolist(), score, len(df_bound),iou_threshold=0.2)
    with tf.Session() as sess:
        out = sess.run([result])
        indexes = np.unique(out[0])
        print(indexes)  
    rs_df = df.iloc[indexes]
    print("write", len(rs_df))
    # rs_df['idx'] = 'None'
    rs_df.to_file(out_file)
    for filename in glob.glob(tmp_file.replace('.shp','.*')):
        print(filename)
        # os.remove(filename)



def nms_shp_one_file(shapefile_path, output_shapefile_path):   

    import geopandas as gpd
    import numpy as np
    import tensorflow as tf
    from shapely.geometry import box

    # Đọc dữ liệu từ shapefile
    gdf = gpd.read_file(shapefile_path)

    # Chuyển đổi các polygon thành bounding box và lưu vào danh sách
    bbox_list = []
    for polygon in gdf['geometry']:
        minx, miny, maxx, maxy = polygon.bounds
        bbox = [minx, miny, maxx, maxy]
        bbox_list.append(bbox)

    # Chuyển danh sách bounding box thành tensor float32
    bbox_tensor = tf.convert_to_tensor(bbox_list, dtype=tf.float32)

    # Lấy danh sách các điểm số confidence
    score = np.array(gdf['score'])

    # Sử dụng hàm non_max_suppression để lọc bounding box dự đoán
    result_indices = tf.image.non_max_suppression(bbox_tensor, score, len(gdf), iou_threshold=0.2)

    # Chọn các bounding box dự đoán dựa trên kết quả của hàm non_max_suppression
    selected_bboxes = tf.gather(bbox_tensor, result_indices)
    selected_scores = tf.gather(score, result_indices)

    # Chuyển các bounding box và scores thành mảng NumPy
    selected_bboxes = selected_bboxes.numpy()
    selected_scores = selected_scores.numpy()

    # Tạo DataFrame mới từ các bounding box đã lọc
    selected_gdf = gpd.GeoDataFrame(geometry=[box(*bbox) for bbox in selected_bboxes], crs=gdf.crs)
    selected_gdf['score'] = selected_scores

    # Lưu kết quả vào shapefile
    selected_gdf.to_file(output_shapefile_path)



# dir_shp = r"/home/skm/SKM_OLD/public/Kolkata_Drone/RS"
# out_file = r"/home/skm/SKM_OLD/public/Kolkata_Drone/RS_ok/ok.shp"

# dir_shp = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/RS/size250/yolov7"
# out_file = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/RS/size250/yolov7/STCD_yolov7_size256_nms.shp"
# nms_shp(dir_shp, out_file)


fp_shp = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/RS/size_128/yolov7/STCD_yolov7_size128.shp"
out_file = r"/home/skm/SKM16/IMAGE/ZZ_ZZ/TauBien/Ship/RS/size_128/yolov7/nms/STCD_yolov7_size128_nms.shp"
nms_shp_one_file(fp_shp, out_file)

