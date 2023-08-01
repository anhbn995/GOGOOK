from geopandas import GeoDataFrame


def get_geospatial_data(gdf: GeoDataFrame):
    minx, miny, maxx, maxy = gdf[~gdf['geometry'].isnull(
    )]['geometry'].total_bounds
    geom_type = gdf.geom_type[0] if gdf.geom_type.size else None
    return {
        'count': len(gdf),
        'crs': str(gdf.crs),
        'geom_type': geom_type,
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
