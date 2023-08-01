from geopandas import GeoDataFrame
from shapely.geometry import shape
from services.vector.io import *


EXCLUDED_TYPES = [
    'MultiplePolygon',
    'MultiplePolyLine',
    'MultiplePolyPoint'
]

VALID_TYPES = [
    sorted(['Point', 'MultiPoint']),
    sorted(['LineString', 'MultiLineString']),
    sorted(['Polygon', 'MultiPolygon'])
]

def validate_gdf(gdf: GeoDataFrame):
    if is_inf_bound(gdf.total_bounds):
        raise Exception('The Shapefile has some features which have invalid geometry (infinity value).')

    if gdf.empty:
        raise Exception('Vector does not contain any features.')

    if len(gdf.type.unique()) > 1:
        is_valid_geom_type = False
        for valid_types in VALID_TYPES:
            if sorted(gdf.type.unique()) == valid_types:
                is_valid_geom_type = True
        if not is_valid_geom_type:
            raise Exception("The Shapefile should only have one geometry type.")
    # for excluded_type in EXCLUDED_TYPES:
    #     if len(gdf[gdf.geometry.type == excluded_type]) > 0:
    #         raise Exception(f'Unsupported {excluded_type}')


def is_valid_geom(geom):
    try:
        shape(geom)
        return 1
    except:
        return 0


def is_inf_bound(bound):
    for corner in bound:
        if corner < -1e100 or corner > 1e100:
            return True
    return False

def read_file(path: str) -> GeoDataFrame:
    ext = path.split('.')[-1]
    reader = {
        'shp': read_shp,
        'kml': read_kml,
        'zip': read_zip_shp,
        'gml': read_gml,
        'geojson': read_geojson,
        'kmz': read_kmz
    }
    return reader.get(ext)(path)


def get_geodataframe(path: str) -> GeoDataFrame:
    gdf = read_file(path)
    gdf = gdf[gdf.is_valid]
    validate_gdf(gdf)
    return gdf


def get_columns(path):
    import fiona
    with fiona.collection(path) as source:
        return list(map(lambda x: {
            "name": x[0],
            "type": x[1].split(':')[0]
        }, dict(source.schema["properties"]).items()))
