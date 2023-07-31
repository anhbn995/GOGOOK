# -*- coding: utf-8 -*-

import multiprocessing
from step_4_adjusting_operators.find_main_axis import find_axis_by_min_degree_error
core_of_computer = multiprocessing.cpu_count()

# Đầu vào: list_error - sai số góc đối với mỗi trục (trong thuật toán tìm trục tốt nhất), 
# error_check - ngưỡng sai số chấp nhận được
# Đầu ra: list_trust - danh sách trục tin tưởng ko tin tưởng (bit =0/1)
def trust_contour(list_error,error_check):
    list_trust = []
    for error in list_error:
        if error <= error_check:
            list_trust.append(1)
        else:
            list_trust.append(0)
    return list_trust

# Đầu vào: list_cntstest - danh sách contour, list_vector - danh sách trục chính, 
# list_trust - danh sách trục tin tưởng (từng phần tử có giá trị 0/1)
# Đầu ra: danh sách contour tin tưởng/ ko tin tưởng cùng với trục chính
def divide_trust_contour(list_cntstest,list_vector,list_translative,list_trust):
    list_contour_trust=[]
    list_contour_not_trust=[]
    list_trust_vector=[]
    list_not_trust_vector=[]
    list_trust_translative=[]
    list_not_trust_translative=[]
    for i in range(len(list_trust)):
        if list_trust[i] ==1:
            list_contour_trust.append(list_cntstest[i])
            list_trust_vector.append(list_vector[i])
            list_trust_translative.append(list_translative[i])
        elif list_trust[i] == 0:
            list_contour_not_trust.append(list_cntstest[i])
            list_not_trust_vector.append(list_vector[i])
            list_not_trust_translative.append(list_translative[i])
    return list_contour_trust, list_contour_not_trust,list_trust_vector,list_not_trust_vector,list_trust_translative,list_not_trust_translative

# Đầu vào: danh sách contour và ngưỡng sai số góc của trục chính
# Đầu ra: danh sách contour tin tưởng/ ko tin tưởng cùng với trục chính
def threshold_contour(list_cntstest,T_SL,list_translative):
    print('threshold_contour')
    list_vector,list_error = find_axis_by_min_degree_error(list_cntstest)
    list_trust = trust_contour(list_error,T_SL)
    # print(list_trust)
    list_contour_trust, list_contour_not_trust,list_trust_vector,list_not_trust_vector,list_trust_translative,list_untrust_translative = divide_trust_contour(list_cntstest,list_vector,list_translative,list_trust)
    return list_contour_trust, list_contour_not_trust,list_trust_vector,list_not_trust_vector,list_trust_translative,list_untrust_translative