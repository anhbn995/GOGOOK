from shapely.geometry import Polygon, LineString, LinearRing, Point, MultiPoint, MultiLineString, GeometryCollection, MultiPolygon, box
import shapely.wkt

def remove_third_dimension(geom):
    if geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        exterior = geom.exterior
        new_exterior = remove_third_dimension(exterior)

        interiors = geom.interiors
        new_interiors = []
        for int in interiors:
            new_interiors.append(remove_third_dimension(int))

        return Polygon(new_exterior, new_interiors)

    elif isinstance(geom, LinearRing):
        return LinearRing([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, LineString):
        return LineString([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, Point):
        return Point([xy[0:2] for xy in list(geom.coords)])

    elif isinstance(geom, MultiPoint):
        points = list(geom.geoms)
        new_points = []
        for point in points:
            new_points.append(remove_third_dimension(point))

        return MultiPoint(new_points)

    elif isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
        new_lines = []
        for line in lines:
            new_lines.append(remove_third_dimension(line))

        return MultiLineString(new_lines)

    elif isinstance(geom, MultiPolygon):
        pols = list(geom.geoms)

        new_pols = []
        for pol in pols:
            new_pols.append(remove_third_dimension(pol))

        return MultiPolygon(new_pols)

    elif isinstance(geom, GeometryCollection):
        geoms = list(geom.geoms)

        new_geoms = []
        for geom in geoms:
            new_geoms.append(remove_third_dimension(geom))

        return GeometryCollection(new_geoms)

    else:
        raise RuntimeError("Currently this type of geometry is not supported: {}".format(type(geom)))

def split_geom_to_grids(geometry, threshold, count=0):
    try:
        bounds = geometry.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if len(shapely.wkt.dumps(geometry).replace(" ", "")) < 15000 or count == 250:
            # either the polygon is smaller than the threshold, or the maximum
            # number of recursions has been reached
            return [geometry]
        if height >= width:
            # split left to right
            a = box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
            b = box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
        else:
            # split top to bottom
            a = box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
            b = box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
        result = []
        for d in (a, b,):
            c = geometry.intersection(d)
            if not isinstance(c, GeometryCollection):
                c = [c]
            for e in c:
                if isinstance(e, (Polygon, MultiPolygon)):
                    result.extend(split_geom_to_grids(e, threshold, count+1))
        if count > 0:
            return result
        # convert multipart into singlepart
        final_result = []
        for g in result:
            g = shapely.wkt.dumps(g.simplify(0.045), rounding_precision=4)
            if isinstance(g, MultiPolygon):
                final_result.extend(g)
            else:
                final_result.append(g)
        return final_result
    except Exception as e:
        print('err1', e)
