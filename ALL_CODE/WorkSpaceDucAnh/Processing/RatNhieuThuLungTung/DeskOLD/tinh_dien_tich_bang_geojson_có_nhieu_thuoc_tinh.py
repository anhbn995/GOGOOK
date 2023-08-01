import rasterio
import rasterio.features
import pandas as pd
import numpy as np
import pyproj


DICT_COLOR = {
    "WORLD_VIEW": {
        'nước': 1,
        'cỏ bụi': 2,
        'có tán': 3,
        'khác': 0
    },
    "SENTINEL_2": {
        'nước': 1,
        'thực vật': 2,
        'khác': 3
    }  
}

def get_pixel_size(img):
    transformer = pyproj.Transformer.from_crs(img.crs.to_epsg(), 3857)

    top_left_point = img.xy(0, 0)
    bottom_right_point = img.xy(img.width, img.height)

    top_left_meter = transformer.transform(top_left_point[1], top_left_point[0])
    bottom_right_meter = transformer.transform(bottom_right_point[1], bottom_right_point[0])

    x_size = abs(bottom_right_meter[0] - top_left_meter[0]) / img.width
    y_size = abs(bottom_right_meter[1] - top_left_meter[1]) / img.height

    return (x_size, y_size)
    

def cal_area(dict_input):
    """
    dict_input: 
        {
        "image": {
            "path": string,
            "type": string, // WORLD_VIEW, SENTINEL_2
                },
        "areas": [
                { 
                "name": string,
                "geometry": GeoJSON 
                }
            ]
        }
        
    """
    img_path = dict_input["image"]["path"]
    type_img = dict_input["image"]["type"]
    list_area = dict_input["areas"]


    # doc img classification
    src = rasterio.open(img_path)
    mask_class = src.read()[0]


    # khoi tao mask vung chay tu 1
    name_area = [x["name"] for x in dict_input['areas']]
    shapes = [(x['geometry'],y) for x,y in zip(dict_input['areas'], range(1,len(list_area)+1))]
    mask_shape =  rasterio.features.rasterize(shapes,
                                    out_shape=(src.height, src.width),
                                    transform=src.transform)
    if np.all((mask_shape == 0)):
        return {"error": "shapefile nam ngoai anh"}
    else:
        values_class = mask_class.flatten()
        zones = mask_shape.flatten()

        df = pd.DataFrame()
        df['zones']= zones
        df['class']= values_class
        df['counter'] = 1

        group_data = df.groupby(['zones','class'])['counter'].sum()
        df_c = group_data.reset_index()
        table = df_c.pivot_table(df_c, index=['zones'], columns=['class'])
        table = table.fillna(0)
        name_colums = [item[1] for item in list(table.columns.values)]
        if len(name_area) < len(np.unique(mask_shape)):
            table = table.iloc[1: , :]
        table = table.T
        table.columns = name_area

        pixel_size = get_pixel_size(src)
        area_unit = pixel_size[0]*pixel_size[1]

        if type_img in ["WORLD_VIEW", "SENTINEL_2"]:
            name_class = list(DICT_COLOR[type_img].keys())
            print(table)
            value_class = [x for x in list(DICT_COLOR[type_img].values())]
            table = table.T
            table.columns = name_colums
            table = table * area_unit

        else:
            raise Exception(f"Không hỗ trợ kiểu ảnh {type_img}")

        rename = dict(zip(value_class,name_class))

        table = table.rename(columns=rename)[name_class]
        result = table.T.to_dict()

        if type_img == "WORLD_VIEW":
            for r in result:
                result[r]["tree"] = result[r]['cỏ bụi'] + result[r]['có tán']
                result[r]["water"] = result[r]["nước"]
                result[r]["other"] = result[r]["khác"]
        elif type_img == "SENTINEL_2":
            for r in result:
                result[r]["tree"] = result[r]['thực vật']
                result[r]["water"] = result[r]["nước"]
                result[r]["other"] = result[r]["khác"]

        list_area = [{"name": a,"statistics": b} for a,b in zip(result.keys(), result.values())]
        dict_rs = {"areas": list_area}
        return dict_rs
if __name__ == "__main__":
    import geopandas as gp
    shape = r"C:/Users/SkyMap/Desktop/ThuDauMot.geojson"
    df_geojson = gp.read_file(shape)
    dict_input = {
        "image": {
            "path": r"Z:\DA\Tmp_Da\2017t12 Thu Dau Mot_0ab.tif",
            "type": "SENTINEL_2"
                },
        "areas": [
                { 
                "name": "Định Hoà",
                "geometry": df_geojson['geometry'][2] 
                }, 
                { 
                "name": "Hoà Phú",
                "geometry": df_geojson['geometry'][5] 
                }, 
                { 
                "name": "Phú Tân",
                "geometry": df_geojson['geometry'][4] 
                }, 
                { 
                "name": "Phú Mỹ",
                "geometry": df_geojson['geometry'][9] 
                }, 
                { 
                "name": "Hiệp Thành",
                "geometry": df_geojson['geometry'][10] 
                }
            ]
        }
    print(cal_area(dict_input))