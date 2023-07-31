# This Python file uses the following encoding: utf-8

import numpy as np
from lib.geometry_calculating import angle_vecto,line_intersection
from numpy.linalg import norm
T_SL = 0.3
T_Projection_Final = 10.0
T_Projection_Final_Untrust = 2/0.3
T_Deviation = 2/0.3
T_Ratio = 0.1
T_Footprint = 0.85
T_Footprint_Untrust = 0.85
T_Auto = 20.0
# Đầu ra: 4 điểm gồm 2 điểm cũ, 2 điểm mới, t_proj: sai số hình chiếu, t_fpr: tỉ lệ cạnh cũ/ cạnh mới
def find_point_base_u_vector(point_1,point_2,trust_vector):
    en_trust_vector = np.array([-trust_vector[1],trust_vector[0]])
    vector_2p = point_2 - point_1
    angle = angle_vecto(vector_2p,trust_vector)
    if 0 <=angle <= 45.0 or 180 >= angle >= 135.0:
        u_vector = trust_vector
    elif 45 < angle < 135:
        u_vector = en_trust_vector
    else:
        u_vector = en_trust_vector
    a_u = u_vector[0]
    b_u = u_vector[1]
    a_n = -u_vector[1]
    b_n = u_vector[0]
        #toa do diem 1
    x_1 = point_1[0]
    y_1 = point_1[1]
    #toa do diem 2
    x_2 = point_2[0]
    y_2 = point_2[1]

    # trung diem hai diem
    x_av = (x_1+x_2)/2.0
    y_av = (y_1+y_2)/2.0
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
    point_3 = np.asarray([x_1_g,y_1_g], dtype = np.float32)
    point_4 = np.asarray([x_2_g,y_2_g], dtype = np.float32)
    vec_p4_p3 = point_4 - point_3
    vec_p2_p1 = point_2 - point_1
    vec_p3_p1 = point_3 - point_1
    t_proj = norm(vec_p3_p1)
    t_fpr = norm(vec_p4_p3)/norm(vec_p2_p1)
    u_vector = None
    return point_3, point_4, t_proj, t_fpr

# Đầu vào: list_array - các điểm trong contour; trust_vector - trục chính 
# Đầu ra: list_angle_define: mảng các cạnh & mỗi cạnh gồm 6 tham số
def cal_list_angle_define(list_array,trust_vector):
    list_angle_define = []
    for i in range(0,len(list_array)-1):
        point_1 = list_array[i]
        point_2 = list_array[i+1]
        point_3, point_4, t_proj, t_fpr = find_point_base_u_vector(point_1,point_2,trust_vector)
        list_angle_define.append([point_1,point_2,point_3, point_4, t_proj, t_fpr])
    point_1_n = list_array[len(list_array)- 1]
    point_2_n = list_array[0]
    point_3_n, point_4_n, t_proj_n, t_fpr_n = find_point_base_u_vector(point_1_n,point_2_n,trust_vector)
    list_angle_define.append([point_1_n,point_2_n,point_3_n, point_4_n, t_proj_n, t_fpr_n])
    return list_angle_define

# Kiểm tra để nắn thẳng theo trục 
def check_strange(angle_define):
    if (angle_define[4] <=T_Projection_Final and angle_define[5] >= T_Footprint) or norm(angle_define[0] - angle_define[1]) <= T_Auto:
        return (angle_define[2],angle_define[3],True)
    else:
        return (angle_define[0],angle_define[1],False)

def check_strange_untrust(angle_define):
    if (angle_define[4] <=T_Projection_Final_Untrust and angle_define[5] >= T_Footprint_Untrust) or norm(angle_define[0] - angle_define[1]) <= T_Auto:
        return (angle_define[2],angle_define[3],True)
    else:
        return (angle_define[0],angle_define[1],False)

# Duyệt qua 2 cạnh liên tiếp và cùng nắn chỉnh
# Đầu vào: list_array - một contour dạng list các điểm, trust_vector: trục chính của contour
# Đầu ra: list_point - contour sau khi được nắn chỉnh (dạng list các điểm)
def extract_step_one_trust(list_array,trust_vector):
    list_point = []
    list_angle_define = cal_list_angle_define(list_array,trust_vector)
    for i in range(len(list_angle_define)-1):
        angle_define_1 = list_angle_define[i]
        angle_define_2 = list_angle_define[i+1]
        point_1,point_2,check_1 = check_strange(angle_define_1)
        point_3,point_4,check_2 = check_strange(angle_define_2)
        vec_1 = point_2 - point_1
        vec_2 = point_4 - point_3
        if angle_vecto(vec_1,vec_2) <= 2 or angle_vecto(vec_1,vec_2)>=178 and (check_1 and check_2):
            list_point.append(point_2)
            list_point.append(point_3)
        else:
            point_xy = line_intersection((point_1,point_2),(point_3,point_4))
            if (np.array(point_xy) == False).any():
                list_point.append(point_2)
                list_point.append(point_3)
            else:
                point_rs = np.asarray(point_xy)
                list_point.append(point_rs)

    angle_define_1 = list_angle_define[len(list_angle_define)-1]
    angle_define_2 = list_angle_define[0]
    point_1,point_2,check_1 = check_strange(angle_define_1)
    point_3,point_4,check_2 = check_strange(angle_define_2)
    vec_1 = point_2 - point_1
    vec_2 = point_4 - point_3
    if (angle_vecto(vec_1,vec_2) <= 2 or angle_vecto(vec_1,vec_2)>=178) and (check_1 and check_2):
        list_point.append(point_2)
        list_point.append(point_3)
    else:
        point_xy = line_intersection((point_1,point_2),(point_3,point_4))
        if (np.array(point_xy) == False).any():
            list_point.append(point_2)
            list_point.append(point_3)
        else:
            point_rs = np.asarray(point_xy)
            list_point.append(point_rs)
        
    return list_point

# Giống như hàm extract_step_one_trust, bỏ qua check ngưỡng
def extract_step_one_untrust(list_array,vector):
    list_point = []
    list_angle_define = cal_list_angle_define(list_array,vector)
    for i in range(len(list_angle_define)-1):
        angle_define_1 = list_angle_define[i]
        angle_define_2 = list_angle_define[i+1]
        point_1,point_2,check_1 = check_strange_untrust(angle_define_1)
        point_3,point_4,check_2 = check_strange_untrust(angle_define_2)
        vec_1 = point_2 - point_1
        vec_2 = point_4 - point_3
        if angle_vecto(vec_1,vec_2) <= 2 or angle_vecto(vec_1,vec_2)>=178 and (check_1 and check_2):
            list_point.append(point_2)
            list_point.append(point_3)
        else:
            point_xy = line_intersection((point_1,point_2),(point_3,point_4))
            if (np.array(point_xy) == False).any():
                list_point.append(point_2)
                list_point.append(point_3)
            else:
                point_rs = np.asarray(point_xy)
                list_point.append(point_rs)

    angle_define_1 = list_angle_define[len(list_angle_define)-1]
    angle_define_2 = list_angle_define[0]
    point_1,point_2,check_1 = check_strange_untrust(angle_define_1)
    point_3,point_4,check_2 = check_strange_untrust(angle_define_2)
    vec_1 = point_2 - point_1
    vec_2 = point_4 - point_3
    if (angle_vecto(vec_1,vec_2) <= 2 or angle_vecto(vec_1,vec_2)>=178) and (check_1 and check_2):
        list_point.append(point_2)
        list_point.append(point_3)
    else:
        point_xy = line_intersection((point_1,point_2),(point_3,point_4))
        if (np.array(point_xy) == False).any():
            list_point.append(point_2)
            list_point.append(point_3)
        else:
            point_rs = np.asarray(point_xy)
            list_point.append(point_rs)
    return list_point