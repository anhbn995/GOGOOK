from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
from step_3_determinate_corners.minE_reduced_search import reduce_search
from step_3_determinate_corners.minE_full_search import full_search
from step_3_determinate_corners.adjusting_corner_by_linear_approx import list_find_all_building_linear
from tqdm import *
core_of_computer = multiprocessing.cpu_count()
T_distance_corner = 1
def determinate_corner_main(contours2,func):
    if func == reduce_search:
        p_cnt = Pool(processes=core_of_computer)
        result = []
        result_1 = p_cnt.imap(partial(func), contours2)
        with tqdm(total=len(contours2)) as pbar:
            for i,contour_temp in tqdm(enumerate(result_1)):
                pbar.update()
                result.append(contour_temp)
        p_cnt.close()
        p_cnt.join()
    elif func == full_search:
        result = full_search(contours2)
    linear_polygon = list_find_all_building_linear(contours2,result,T_distance_corner)
    # linear_polygon = None
    # linear_polygon = result
    return(result,linear_polygon)