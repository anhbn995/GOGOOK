import json
from geopandas import GeoDataFrame
import geopandas as gpd
from services.db import engine
from shapely.geometry import Point
from functools import partial
import pyproj
from shapely.ops import transform
from services.vector.io import write_shp_to_zip
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon


GEOM_TYPE = {
    'MULTIPOLYGON': MultiPolygon,
    'MULTILINESTRING': MultiLineString,
    'MULTIPOINT': MultiPoint
}

def convert_to_postgis(gdf: GeoDataFrame, table_name: str, table_schema: str, fields=None):
    epsg = 4326
    gdf_columns = gdf.columns.to_list()
    con = engine.connect()
    try:
        geometry_col_info = con.execute(
            f"SELECT type, srid FROM geometry_columns WHERE f_table_name = '{table_name}' AND f_table_schema = '{table_schema}' AND f_geometry_column = 'geometry'"
        ).fetchone()
        if geometry_col_info:
            geometry_type = geometry_col_info[0]
            epsg = geometry_col_info[1] if geometry_col_info[1] != 0 else 4326

            if 'MULTI' in geometry_type:
                gdf['geometry'] = [GEOM_TYPE[geometry_type]([feature]) if not isinstance(feature, GEOM_TYPE[geometry_type]) else feature for feature in gdf["geometry"]]

        if not gdf.crs.to_epsg():
            gdf = gdf.to_crs(epsg=epsg)

        table_columns = list(map(lambda rows: rows[0], con.execute(
            f"SELECT column_name FROM information_schema.columns \
            WHERE table_name = '{table_name}' \
            and table_schema = '{table_schema}' \
            and column_name != 'id'"
        ).fetchall()))
        if len(table_columns) > 0:
            included_columns = fields or table_columns
            dropped_columns = list(
                filter(lambda i: i not in table_columns or i not in included_columns, gdf_columns))
            dropped_columns.remove('geometry')
            gdf.drop(columns=dropped_columns, inplace=True)
            con.execute(f'TRUNCATE TABLE "{table_schema}"."{table_name}"')
        gdf.to_postgis(table_name, con, if_exists='append', schema=table_schema)
        con.execute(
            f'CREATE INDEX {table_schema}_{table_name}_geometry_idx ON "{table_schema}"."{table_name}" USING GIST (geometry)')
    except Exception as e:
        raise Exception(str(e).split('CONTEXT')[0].split('DETAIL')[0])
    finally:
        con.close()


def inspect(tables, lat, lng, buffer_length):
    results = []
    with engine.connect() as con:
        for table in tables:
            schema, name = table.split('.')
            row = con.execute(
                f"select *, st_asgeojson(st_transform(geometry,4326)) as geometry from \"{schema}\".{name} \
                where st_intersects(ST_Buffer(st_transform(st_setsrid(ST_GeomFromText('POINT({lng} {lat})'),4326),3857), {buffer_length}, 'quad_segs=8'), st_transform(geometry::geometry,3857))"
            ).first()
            if not row:
                results.append(None)
                continue
            result = json.loads(json.dumps({key: value for (
                key, value) in row.items()}, indent=4, sort_keys=True, default=str))
            result['geometry'] = json.loads(result['geometry'])
            results.append(result)
    return results


def buffer_point(lat, lon, length):
    aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    project = partial(
        pyproj.transform,
        pyproj.Proj(aeqd_proj.format(lat=lat, lon=lon)),
        pyproj.Proj('+proj=longlat +datum=WGS84'))
    buf = Point(0, 0).buffer(length)  # distance in metres
    return {
        "type": "Polygon",
        'coordinates': [list(map(lambda coords: list(coords), transform(project, buf).exterior.coords[:]))]
    }


def create_filters_query(filters):
    if not filters:
        return ''
    combining_filter = ' or ' if filters['combiningFilter']['value'] == 'any' else ' and '
    filter_list = []
    for filter in filters['filters']:
        if filter['operator'] == '==':
            filter['operator'] = '='
        if filter['operator'] in ['in', '!in']:
            values = [x if filter['dataType'] ==
                      'number' else f"'{x}'" for x in filter['value'].split(',')]
            filter_value = (', ').join(values)

        else:
            filter_value = filter['value'] if filter[
                'dataType'] == 'number' else f"'{filter['value']}'"
        filter_value = f'({filter_value})'
        filter_str = f"\"{filter['property']}\" {filter['operator']} {filter_value}"
        if filters['combiningFilter']['value'] == 'none':
            filter_str = 'not ' + filter_str
        filter_list.append(filter_str)
    return f' where {combining_filter.join(filter_list)}'


def read_postgis(sql) -> GeoDataFrame:
    with engine.connect() as con:
        gdf = gpd.read_postgis(sql, con, geom_col='geometry')
    from datetime import date
    for c in gdf.columns:
        if gdf[c].dtype == 'datetime64[ns]':
            gdf[c] = gdf[c].astype(str)
            continue
        if gdf[c].dtype != object:
            continue
        first_value_idx = gdf[c].first_valid_index()
        if first_value_idx == None:
            continue
        if isinstance(gdf[c].loc[first_value_idx], date):
            gdf[c] = gdf[c].astype(str)
    return gdf


def to_file(path: str, name: str, table: str, filters=None):
    filters = json.loads(filters) if filters else None
    filters_query = create_filters_query(filters)
    table_schema, table_name = table.split('.')
    sql = f'SELECT * FROM "{table_schema}"."{table_name}" {filters_query}'
    gdf = read_postgis(sql)
    if 'updated_at' in gdf.columns and 'created_at' in gdf.columns:
        gdf.drop(columns=['updated_at', 'created_at'], inplace=True)
    gdf.to_file(path)
    return write_shp_to_zip(path, arcname=name, remove_shp=True)
