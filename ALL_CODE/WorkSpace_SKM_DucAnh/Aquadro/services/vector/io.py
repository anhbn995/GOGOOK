from glob import glob
from lib2to3.pgen2 import driver
import os
import zipfile
from geopandas import read_file, GeoDataFrame, io


def read_shp(path: str) -> GeoDataFrame:
    return read_file(path, encoding='utf-8')


def read_kml(path: str) -> GeoDataFrame:
    io.file.fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
    return read_file(path, driver='KML', encoding='utf-8')


def read_kmz(path: str) -> GeoDataFrame:
    io.file.fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
    return read_file(path, driver='LIBKML', encoding='utf-8')


def read_zip_shp(path: str) -> GeoDataFrame:
    return read_file(f'zip://{path}', encoding='utf-8')


def read_geojson(path: str) -> GeoDataFrame:
    return read_file(path, driver='GeoJSON', encoding='utf-8')


def read_gml(path: str) -> GeoDataFrame:
    io.file.fiona.drvsupport.supported_drivers['GML'] = 'rw'
    return read_file(path, driver='GML', encoding='utf-8')


def write_shp_to_zip(path, arcname=None, remove_shp=False):
    output_path = path.replace('.shp', '.zip')
    with zipfile.ZipFile(output_path, "w") as zip:
        for f in glob(path.replace(".shp", ".*")):
            if f.endswith('.zip'):
                continue
            zip.write(
                f, f"{arcname}{os.path.splitext(f)[1]}" if arcname else os.path.basename(f))
            if remove_shp:
                os.remove(f)
    return output_path
