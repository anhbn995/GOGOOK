import numpy as np

def convert_obj(img):
    img1 = img/255
    img1[img1==1]=2
    img1[img1==0]=1
    img1[img1==2]=0
    return img1

def rm_polygon_err(list_polygon):
    list_poly_good =[]
    for polygon in list_polygon:
        if len(polygon) >= 3:
            list_poly_good.append(polygon)
    return list_poly_good

def contour_to_polygon(contour):
    list_point = []
    try:
        for point in contour:
            [x,y] = point[0]
            point_in_polygon = (x,y)
            list_point.append(point_in_polygon)
        # print(contour[0])
        [x,y] = contour[0][0]
        point_in_polygon = (x,y)
        list_point.append(point_in_polygon)
        poly = tuple(list_point)
    except Exception:
        poly = ()
    return poly

def list_contour_to_list_polygon(list_contour):
    list_polygon = []
    for contour in list_contour:
        poly = contour_to_polygon(contour)
        list_polygon.append(poly)
    return list_polygon

def point_to_geopoint(point, geotransform):
    topleftX = geotransform[0]
    topleftY = geotransform[3]
    XRes = geotransform[1]
    YRes = geotransform[5]
    geopoint = (topleftX + point[0] * XRes, topleftY + point[1] * YRes)
    return geopoint

def polygon_to_geopolygon(polygon, geotransform):
    temp_geopolygon = []
    for point in polygon:
        geopoint = point_to_geopoint(point, geotransform)
        temp_geopolygon.append(geopoint)
    geopolygon = tuple(temp_geopolygon)
    return geopolygon

def list_polygon_to_list_geopolygon(list_polygon, geotransform):
    list_geopolygon = []
    for polygon in list_polygon:
        geopolygon = polygon_to_geopolygon(polygon, geotransform)
        list_geopolygon.append(geopolygon)
    # p_geocal = Pool(processes=core_of_computer)
    # result = p_geocal.map(partial(polygon_to_geopolygon,geotransform=geotransform), list_polygon)
    # p_geocal.close()
    # p_geocal.join()
    # list_geopolygon = result
    return list_geopolygon