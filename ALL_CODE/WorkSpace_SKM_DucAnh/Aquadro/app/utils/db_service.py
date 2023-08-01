import psycopg2
from config.app import DB_URL
from datetime import date


def get_geometry_from_aoi(aoi):
    if len(aoi.split('_')) > 1:
        aoi_id = aoi.split('_')[1]
        url = DB_URL[aoi.split('_')[0]]
    else:
        aoi_id = aoi
        url = DB_URL['monitoring']

    conn = psycopg2.connect(url)
    cursor = conn.cursor()
    cursor.execute("select wkt_geom from aois where id = %s", (aoi_id,))
    data = cursor.fetchone()[0]
    conn.close()
    return data


def get_geometry_from_field(aoi_ids):
    aoi_ids = str(aoi_ids)
    data_aoi = aoi_ids.split('_')
    if len(data_aoi) > 1:
        aoi_ids = data_aoi[1].split(',')
        url = DB_URL[data_aoi[0]]
    else:
        aoi_ids = data_aoi[0].split(',')
        url = DB_URL['monitoring']

    conn = psycopg2.connect(url)
    cursor = conn.cursor()
    cursor.execute("select st_astext(st_union(geometry::geometry)) from fields where id in %s", (tuple(aoi_ids),))
    data = cursor.fetchone()[0]
    conn.close()
    return data


def get_images(aoi_id):
    year = date.today().year - 1
    conn = psycopg2.connect(DB_URL['monitoring'])
    cursor = conn.cursor()
    cursor.execute(f"select date, index_statistics "
                   f"from images where field_id = {aoi_id} and image_source = 'SENTINEL2_L2A' and index_statistics->'NDVI' is not null "
                   f"and field_cloud_cover < 0.1 and date between '{year}-01-01' and '{year}-12-31'")
    data = cursor.fetchall()
    conn.close()
    return data
