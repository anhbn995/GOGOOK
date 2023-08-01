from pyproj import Geod
from rio_tiler.models import ImageData
from rasterio.io import MemoryFile
import io


def hex_2_rgb(hex):
    hex = hex.replace('#', '')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, int(hlen // 3)))


def rgb_2_hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def calculate_area(boundary):
    geod = Geod(ellps="WGS84")
    area, _ = geod.geometry_area_perimeter(boundary)
    return abs(area)


def geometry_contains(geom_a, geom_b):
    # return True if geometry B is completely inside geometry A
    return geom_a.contains(geom_b)


def file_from_image_data(data: ImageData):
    profile = {
        'driver': 'GTiff',
        'dtype': data.data.dtype,
        'nodata': 0,
        'width': data.width,
        'height': data.height,
        'count': 1,
        'crs': data.crs,
        'transform': data.transform,
    }

    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(data.data)

        return io.BytesIO(memfile.getbuffer())
