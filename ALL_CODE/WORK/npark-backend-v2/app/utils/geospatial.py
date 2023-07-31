import geopandas as gpd


def get_geospatial_data(file_path):
    data_frame = gpd.read_file(file_path)
    minx, miny, maxx, maxy = data_frame[~data_frame['geometry'].isnull()]['geometry'].total_bounds
    return {
        'srs': str(data_frame.crs),
        'bbox': {
            'type': 'Polygon',
            'coordinates': [
                [
                    [minx, miny],
                    [minx, maxy],
                    [maxx, maxy],
                    [maxx, miny],
                    [minx, miny]
                ]
            ]
        }
    }


if __name__ == "__main__":
    data = get_geospatial_data('/home/nghipham/A3_Farm_Boundary.shp')
    print(data)
