import shapely.wkt
from app.utils.db_service import get_geometry_from_field, get_geometry_from_aoi
from rasterio.warp import transform_geom
from rasterio.rio.helpers import coords
from rasterio.transform import rowcol
from affine import Affine
from rasterio import MemoryFile
from rasterio.mask import mask
import json
import geopandas as gpd

import numpy as np


def crop_tile(tile_data, fields=None, aoi=None, geometry=None):
    if fields:
        geometry = shapely.wkt.loads(get_geometry_from_field(fields))
    elif aoi:
        geometry = shapely.wkt.loads(get_geometry_from_aoi(aoi))
    geopd = gpd.GeoSeries([geometry], crs={'init': 'epsg:4326'})
    geopd = geopd.to_crs(crs=tile_data.crs.data)
    coords = [json.loads(geopd.to_json())['features'][0]['geometry']]

    profile = {
        'driver': 'GTiff',
        'dtype': tile_data.data.dtype,
        'nodata': 0,
        'width': tile_data.width,
        'height': tile_data.height,
        'count': 1,
        'crs': tile_data.crs,
        'transform': tile_data.transform,
    }
    profile.update(transform=tile_data.transform, driver='GTiff', height=tile_data.height, width=tile_data.width)

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            if len(tile_data.data) > 1:
                dataset.write(np.array([tile_data.data[0]]))
            else:
                dataset.write(tile_data.data)
            tile, transform = mask(dataset=dataset, shapes=coords)

    return tile


def generate_cutline(geometry, crs, transform, width, height):
    geom_type = geometry.type
    if geom_type not in ["Polygon", "MultiPolygon"]:
        raise Exception("Invalid geometry type")
    geometry = transform_geom("epsg:4326", crs, geometry)
    polys = []
    geom = ([geometry["coordinates"]] if geom_type == "Polygon" else geometry["coordinates"])

    for p in geom:
        xs, ys = zip(*coords(p))
        src_y, src_x = rowcol(transform, xs, ys)
        src_x = [max(0, min(width, x)) for x in src_x]
        src_y = [max(0, min(height, y)) for y in src_y]
        poly = ", ".join(
            [f"{x} {y}" for x, y in list(zip(src_x, src_y))])
        polys.append(f"(({poly}))")

    str_poly = ",".join(polys)
    return f"POLYGON {str_poly}" if geom_type == "Polygon" else f"MULTIPOLYGON ({str_poly})"


def get_projection_information(cog_dataset=None, metadata=None):
    if metadata is not None:
        return {
            'crs': f"epsg:{metadata['properties']['proj:epsg']}",
            'transform': Affine(*metadata['assets']['B04']['proj:transform'][0:6]),
            'width': metadata['assets']['B04']['proj:shape'][1],
            'height': metadata['assets']['B04']['proj:shape'][0]
        }

    if cog_dataset is not None:
        return {
            'crs': cog_dataset.crs,
            'transform': cog_dataset.transform,
            'width': cog_dataset.width,
            'height': cog_dataset.height
        }


def generate_vrt(cog_dataset=None, metadata=None, geometry=None, aois=None):
    if aois:
        geometry = shapely.wkt.loads(get_geometry_from_field(aois))
    else:
        if geometry is None:
            return None

    args = get_projection_information(cog_dataset=cog_dataset, metadata=metadata)
    args['geometry'] = geometry
    return {
        'cutline': generate_cutline(**args),
        'CUTLINE_ALL_TOUCHED': True
    }
