import numpy as np
from lib.geometry_calculating import angle_vecto,line_intersection
from numpy.linalg import norm
from lib.convert_datatype import contour_to_list_array,list_array_to_contour
from step_4_adjusting_operators.extract_contour_by_vector import extract_step_one_trust
import multiprocessing 
core_of_computer = multiprocessing.cpu_count()
from multiprocessing.pool import Pool
from functools import partial
T_SL = 0.3
T_Projection_Final = 2/0.3
T_Deviation = 2/0.3
T_Ratio = 0.1
T_Footprint = 0.85

def strange_contour_trust_main(list_contour_trust,list_trust_vector):
    print('strange_contour_trust_main')
    list_i = list(range(len(list_contour_trust)))
    p_strange = Pool(processes=core_of_computer)
    result = p_strange.map(partial(strange_contour_trust_pool,list_contour_trust=list_contour_trust,list_trust_vector=list_trust_vector), list_i)
    p_strange.close()
    p_strange.join()
    list_contour_strange_trust = result
    return list_contour_strange_trust

def strange_contour_trust_pool(i,list_contour_trust,list_trust_vector):
    try:
        list_array = contour_to_list_array(list_contour_trust[i])
        list_array_rs = extract_step_one_trust(list_array,list_trust_vector[i])

        contour_rs = list_array_to_contour(list_array_rs)
    except Exception: 
        contour_rs = list_contour_trust[i]
    return contour_rs
    
