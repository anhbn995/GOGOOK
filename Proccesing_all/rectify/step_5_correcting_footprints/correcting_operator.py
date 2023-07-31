# This Python file uses the following encoding: utf-8
from lib.geometry_calculating import vector_of_degrees,angle_vecto,remove_duple_item,line_intersection,area_triangle,projection_point_on_line
from numpy.linalg import norm
import numpy as np
import cv2
from lib.convert_datatype import list_array_to_contour,contour_to_list_array,convert_numpy_to_list
from shapely.geometry import Polygon,Point,MultiPolygon
# input : polygon, kiểu dũ liệu list các aray
# vấn đề: có 3 canh liên tục. cạnh ở giữa vuông góc với hai cạnh còn lại, tỉ lệ ạnh giũa/(tổng chiều dài hai cạnh còn lại)< 0,1 và chiều dài cạnh giữa nhỏ hơn 1/0,3 pixel
# xử lý: hợp nhất 3 cạnh thành một cạnh. cạnh này dịch lên hay xuống dựa theo cạnh dài hơn
def fix_bug_001(contour):
    list_attack = contour_to_list_array(contour)
    # list_attack.extend([list_point[0],list_point[1], list_point[2]])
    i=0
    while(i<len(list_attack)+1):
        point_1 = list_attack[(i%len(list_attack))]
        point_2 = list_attack[((i+1)%len(list_attack))]
        point_3 = list_attack[((i+2)%len(list_attack))]
        point_4 = list_attack[((i+3)%len(list_attack))]
        vec_1 = point_2 - point_1
        vec_2 = point_3 - point_2
        vec_3 = point_4 - point_3
        scale_dived = norm(vec_2)/(norm(vec_3)+norm(vec_1))
        scale_length = norm(vec_2)
        if (angle_vecto(vec_1,vec_3) < 2.0) and (scale_dived < 0.2) and (scale_length < 10.0) and (88 < angle_vecto(vec_1,vec_2) < 92)or (angle_vecto(vec_1,vec_3) < 2.0) and (scale_length < 5.0) and (88 < angle_vecto(vec_1,vec_2) < 92):
            u_vector = vec_1
            scale = norm(vec_3)/(norm(vec_3)+norm(vec_1))
            point_base = point_2 + vec_2*scale
            a_u = u_vector[0]
            b_u = u_vector[1]
            a_n = -u_vector[1]
            b_n = u_vector[0]
                    #toa do diem 1
            x_1 = point_1[0]
            y_1 = point_1[1]
            #toa do diem 2
            x_2 = point_4[0]
            y_2 = point_4[1]

            # trung diem hai diem
            x_av = point_base[0]
            y_av = point_base[1]
                # phuong trinh phap tuyen di qua diem 1
            a_n_1 = a_u
            b_n_1 = b_u
            c_n_1 = -(a_n_1*x_1 + b_n_1*y_1)
            # phuong trinh phap tuyen di qua diem 2
            a_n_2 = a_u
            b_n_2 = b_u
            c_n_2 = -(a_n_2*x_2 + b_n_2*y_2)
            # phuong trinh chi phuong di qua trung diem
            a_u_av = a_n
            b_u_av = b_n
            c_u_av = -(a_u_av*x_av + b_u_av*y_av)
            # toa do giao diem 1:
            x_1_g = (c_u_av*b_n_1 - c_n_1*b_u_av)/(a_n_1*b_u_av - a_u_av*b_n_1)
            y_1_g = (c_u_av*a_n_1 - c_n_1*a_u_av)/(b_n_1*a_u_av - b_u_av*a_n_1)
            # toa do giao diem 2:
            x_2_g = (c_u_av*b_n_2 - c_n_2*b_u_av)/(a_n_2*b_u_av - a_u_av*b_n_2)
            y_2_g = (c_u_av*a_n_2 - c_n_2*a_u_av)/(b_n_2*a_u_av - b_u_av*a_n_2)
            #    result
            point_rs1 = np.asarray([x_1_g,y_1_g], dtype = np.float32)
            point_rs2 = np.asarray([x_2_g,y_2_g], dtype = np.float32)
            list_attack[(i%len(list_attack))] = point_rs1
            list_attack[((i+3)%len(list_attack))]= point_rs2
            ind2remove = [((i+1)%len(list_attack)),((i+2)%len(list_attack))]

            list_attack = [x for j,x in enumerate(list_attack) if j not in ind2remove]
            # del list_attack[[((i+1)%len(list_attack)),((i+2)%len(list_attack))]]
            i = i
        else:
            i = i+1
    result_contour = list_array_to_contour(list_attack)
    return result_contour
    # lỗi hình tam giác
def fix_bug_002(contour):
    list_attack = contour_to_list_array(contour)
    i=0
    while(i<len(list_attack)+1):
        point_1 = list_attack[(i%len(list_attack))]
        point_2 = list_attack[((i+1)%len(list_attack))]
        point_3 = list_attack[((i+2)%len(list_attack))]
        point_4 = list_attack[((i+3)%len(list_attack))]
        vec_1 = point_2 - point_1
        vec_2 = point_3 - point_2
        vec_3 = point_4 - point_3
        if (88 < angle_vecto(vec_1,vec_3) < 92) and (88<(angle_vecto(vec_1,vec_2) + angle_vecto(vec_2,vec_3))< 92):
            scale_dived = norm(vec_2)/(norm(vec_3)+norm(vec_1))
            scale_length = norm(vec_2)
            point_5 = line_intersection((point_1,point_2),(point_3,point_4))
            triangle_big = area_triangle(point_1,point_5,point_4)
            triangle_small = area_triangle(point_2,point_5,point_3)
            scale_triangle = triangle_small/triangle_big
            if (scale_dived < 0.3) and (scale_length < 50.0) and  (scale_triangle < 0.5)and(triangle_small<200):
                list_attack[((i+1)%len(list_attack))] = point_5
                del list_attack[((i+2)%len(list_attack))]
        i = i+1
    result_contour = list_array_to_contour(list_attack)
    return result_contour
    #lỗi xén bớt hình chữ nhật
def fix_bug_003(contour):
    area_contour = cv2.contourArea(contour)
    list_attack = contour_to_list_array(contour)
    i=0
    while(i<len(list_attack)+1):
        point_1 = list_attack[(i%len(list_attack))]
        point_2 = list_attack[((i+1)%len(list_attack))]
        point_3 = list_attack[((i+2)%len(list_attack))]
        point_4 = list_attack[((i+3)%len(list_attack))]
        vec_1 = point_2 - point_1
        vec_2 = point_3 - point_2
        vec_3 = point_4 - point_3
        scale_dived = norm(vec_2)/(norm(vec_3)+norm(vec_1))
        scale_length = norm(vec_2)
        if norm(vec_1)>norm(vec_3):
            point_5 = projection_point_on_line(point_1,point_2,point_4)
            rectangle_area = norm(vec_2)*norm(vec_3)
        else:
            point_5 = projection_point_on_line(point_3,point_4,point_1)
            rectangle_area = norm(vec_2)*norm(vec_3)
        scale_rectangle = rectangle_area/area_contour
        if ((angle_vecto(vec_1,vec_3)>178) and (scale_dived < 0.2) and (scale_length < 10.0) and (scale_rectangle<0.02) and (88 < angle_vecto(vec_1,vec_2)< 92) or (rectangle_area<50) ):
            list_attack[((i+1)%len(list_attack))] = point_5
            del list_attack[((i+2)%len(list_attack))]
            i=i
        else:
            i = i+1
    result_contour = list_array_to_contour(list_attack)
    return result_contour

        #lỗi xén bớt hình thang
def fix_bug_004(contour):
    area_contour = cv2.contourArea(contour)
    list_attack = contour_to_list_array(contour)
    i=0
    while(i<len(list_attack)+1):
        point_1 = list_attack[(i%len(list_attack))]
        point_2 = list_attack[((i+1)%len(list_attack))]
        point_3 = list_attack[((i+2)%len(list_attack))]
        point_4 = list_attack[((i+3)%len(list_attack))]
        vec_1 = point_2 - point_1
        vec_2 = point_3 - point_2
        vec_3 = point_4 - point_3
        scale_dived = norm(vec_2)/(norm(vec_3)+norm(vec_1))
        scale_length = norm(vec_2)
        if norm(vec_1)>norm(vec_3):
            point_5 = projection_point_on_line(point_1,point_2,point_4)
            quadrilateral_area = norm(point_4-point_5)*(norm(vec_3)+norm(point_2-point_5))/2
        else:
            point_5 = projection_point_on_line(point_3,point_4,point_1)
            quadrilateral_area = norm(point_1-point_5)*(norm(vec_1)+norm(point_3-point_5))/2
        scale_quadrilateral = quadrilateral_area/area_contour
        if ((angle_vecto(vec_1,vec_3)>178) and (scale_dived < 0.2) and (scale_length < 10.0) and (scale_quadrilateral< 0.02) or (quadrilateral_area<50)):
            list_attack[((i+1)%len(list_attack))] = point_5
            del list_attack[((i+2)%len(list_attack))]
        i = i+1
    result_contour = list_array_to_contour(list_attack)
    return result_contour

    # trương hợp tạo thành tam giác thừa
def fix_bug_005(contour):
    list_attack = contour_to_list_array(contour)
    i = 0
    while(i<len(list_attack)+1):
        point_1 = list_attack[(i%len(list_attack))]
        point_2 = list_attack[((i+1)%len(list_attack))]
        point_3 = list_attack[((i+2)%len(list_attack))]
        point_4 = list_attack[((i+3)%len(list_attack))]
        vec_1 = point_2 - point_1
        vec_2 = point_4 - point_3
        if 88<angle_vecto(vec_1,vec_2)<92:
            point_g = line_intersection((point_1,point_2),(point_3,point_4))
            d1=norm(point_1-point_g)
            d2=norm(point_2-point_g)
            d3=norm(point_3-point_g)
            d4=norm(point_4-point_g)
            dx=norm(point_1-point_2)
            dy = norm(point_3-point_4)
            eps1=(d1+d2)/dx
            eps2=(d3+d4)/dy
            area_small = area_triangle(point_2,point_3,point_g)
            area_big = area_triangle(point_1,point_4,point_g)
            epsilon = area_small/area_big
            if (0.99 < eps1 < 1.01) and (0.99 < eps2 < 1.01) and (epsilon < 0.1):# and (d1 < d) and (d3 < d) and (d4 < d) and (d2 < d):
                list_attack[((i+1)%len(list_attack))] = point_g
                del list_attack[((i+2)%len(list_attack))]
        i=i+1
    result_contour = list_array_to_contour(list_attack)
    return result_contour
 # hình chữ nhật nhỏ chéo nhau
def fix_bug_006(contour):
    list_attack = contour_to_list_array(contour)
    i = 0
    while(i<len(list_attack)+1):
        point_1 = list_attack[(i%len(list_attack))]
        point_2 = list_attack[((i+1)%len(list_attack))]
        point_3 = list_attack[((i+2)%len(list_attack))]
        point_4 = list_attack[((i+3)%len(list_attack))]
        point_5 = list_attack[((i+4)%len(list_attack))]
        vec_1 = point_2 - point_1
        vec_2 = point_3 - point_2
        vec_3 = point_4 - point_3
        vec_4 = point_5 - point_4

        if (88<angle_vecto(vec_1, vec_4)<92) and (angle_vecto(vec_1, vec_3)>178) and (angle_vecto(vec_2, vec_4)>178):
            point_g = line_intersection((point_1,point_2),(point_4,point_5))
            d1=norm(point_1-point_g)
            d2=norm(point_2-point_g)
            d3=norm(point_3-point_2)
            d4=norm(point_4-point_3)
            d5 = norm(point_4-point_g)
            d6 = norm(point_5-point_g)
            dx=norm(point_1-point_2)
            dy = norm(point_5-point_4)
            eps1=(d1+d2)/dx
            eps2=(d5+d6)/dy
            if (0.99 < eps1 < 1.01) and (0.99 < eps2 < 1.01):# and (d1 < d) and (d3 < d) and (d4 < d) and (d2 < d):
                rec_small = d2*d5
                rec_big = d1*d6
                epsilon = rec_small/rec_big
                if epsilon < 0.3 and (rec_small<200):
                    list_attack[((i+1)%len(list_attack))] = point_g
                    ind2remove = [((i+2)%len(list_attack)),((i+3)%len(list_attack))]
                    list_attack = [x for j,x in enumerate(list_attack) if j not in ind2remove]
        i=i+1
    result_contour = list_array_to_contour(list_attack)
    return result_contour
# loi tam giac thua ra ngoai khong cheo nhau
def fix_bug_007(contour):
    area_contour = cv2.contourArea(contour)
    list_attack = contour_to_list_array(contour)
    i=0
    while(i<len(list_attack)+1):
        point_1 = list_attack[(i%len(list_attack))]
        point_2 = list_attack[((i+1)%len(list_attack))]
        point_3 = list_attack[((i+2)%len(list_attack))]
        point_4 = list_attack[((i+3)%len(list_attack))]
        vec_1 = point_2 - point_1
        vec_2 = point_3 - point_2
        vec_3 = point_4 - point_3
        if ((85 < angle_vecto(vec_1,vec_3) < 95) and (max(angle_vecto(vec_1,vec_2),angle_vecto(vec_2,vec_3))>90)):

            point_g = line_intersection((point_1,point_2),(point_3,point_4))
            triangle_small = area_triangle(point_2,point_g,point_3)
            scale_triangle = triangle_small/area_contour
            if ((scale_triangle < 0.1)): #and (triangle_small<500)):
                list_attack[((i+1)%len(list_attack))] = point_g
                del list_attack[((i+2)%len(list_attack))]
        i = i+1
    result_contour = list_array_to_contour(list_attack)
    return result_contour

def remove_point_on_segment_0(contour):#p1,p2,p3 la 3 diem lien tiep
    list_array = contour_to_list_array(contour)
    i = 1
    while (i<len(list_array)+2):
        p1=list_array[(i-1)%len(list_array)]
        p2=list_array[i%len(list_array)]
        p3=list_array[(i+1)%len(list_array)]

        #cac vecto lien tiep
        vec1=p2-p1
        vec2=p3-p2

        angle=angle_vecto(vec1,vec2)

        if angle < 2:
            del list_array[i%len(list_array)]
        else:
            i = i+1
    result = list_array_to_contour(list_array)
    return result

def remove_point_on_segment(contour):#p1,p2,p3 la 3 diem lien tiep
    list_array = contour_to_list_array(contour)
    i = 1
    while (i<len(list_array)+2):
        p1=list_array[(i-1)%len(list_array)]
        p2=list_array[i%len(list_array)]
        p3=list_array[(i+1)%len(list_array)]

        #cac vecto lien tiep
        vec1=p2-p1
        vec2=p3-p2

        angle=angle_vecto(vec1,vec2)

        if angle > 178:
            del list_array[i%len(list_array)]
        else:
            i = i+1
    result = list_array_to_contour(list_array)
    return result

def fix_list_polygon_union(list_polygon):
    poly_union = Polygon(list_polygon[0])
    for i in range(1,len(list_polygon)):
        polyi = Polygon(list_polygon[i])
        poly_union = poly_union.union(polyi)
    list_polygon_fixed =  []
    for poly in poly_union:
        polygon = tuple(list(poly.exterior.coords))
        list_polygon_fixed.append(polygon)
    return list_polygon_fixed