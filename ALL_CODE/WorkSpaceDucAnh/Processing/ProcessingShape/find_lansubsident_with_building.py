import geopandas as gpd

# fp_building = r'/home/skm/SKM16/Data/Uzabekittan/Test/in/polygon_nho.shp'
# fp_land_supsident = r'/home/skm/SKM16/Data/Uzabekittan/Test/in/point_smalll.shp'


fp_building = r'/home/skm/SKM16/Data/Uzabekittan/RS_ALL_BUILDING.shp'
fp_land_supsident = r'/home/skm/SKM16/Data/Uzabekittan/PointlandSupsident/Land_sub_shp/Land_subsidence_v3.shp'

# Đường dẫn đến tệp tin Shapefile chứa toàn polygon
polygon_shapefile_path = fp_building

# Đường dẫn đến tệp tin Shapefile chứa các điểm
point_shapefile_path = fp_land_supsident

# Đọc tệp tin Shapefile chứa toàn polygon vào GeoDataFrame
polygon_gdf = gpd.read_file(polygon_shapefile_path)
polygon_gdf['id'] = list(range(0, len(polygon_gdf)))
print(polygon_gdf.columns)
# Đọc tệp tin Shapefile chứa các điểm vào GeoDataFrame
point_gdf = gpd.read_file(point_shapefile_path)


# Thực hiện phép giao giữa hai lớp đối tượng
# intersection_gdf = gpd.overlay(point_gdf, polygon_gdf,how='intersection')


# Tạo vùng buffer với bán kính 5 mét cho các điểm
buffer_distance = 0.000025  # Bán kính buffer là 5 mét
point_gdf['geometry'] = point_gdf.geometry.buffer(buffer_distance)

# # Thực hiện phép giao giữa vùng buffer và toàn polygon
# intersection_gdf = gpd.sjoin(point_gdf, polygon_gdf, how='inner', op='intersects')
intersection_gdf = gpd.sjoin(polygon_gdf, point_gdf, how='inner', op='intersects')
# Lấy lại các điểm gốc
original_points_gdf = intersection_gdf#[point_gdf.columns]
result_df = original_points_gdf.drop_duplicates(subset='id')

updated_data_b = polygon_gdf.merge(result_df.drop(columns=['geometry']), on='id', how='left')
# updated_data_b_ = gpd.GeoDataFrame(updated_data_b)
# Lưu kết quả vào tệp tin Shapefile mới
output_shapefile_path = r'/home/skm/SKM16/Data/Uzabekittan/Test/out/RS.shp'
updated_data_b.to_file(output_shapefile_path)