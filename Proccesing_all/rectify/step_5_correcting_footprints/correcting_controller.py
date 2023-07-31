from step_5_correcting_footprints.correcting_operator import fix_bug_001,fix_bug_006,fix_bug_003,remove_point_on_segment_0,fix_bug_002,remove_point_on_segment,fix_bug_007
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
core_of_computer = multiprocessing.cpu_count()
def correcting_main(list_polygon):
    print('correcting_main')
    # list_polygon_new = []
    # for polygon in list_polygon:
    #     try:
    #         polygon_new = fix_bug_001(polygon)
    #         polygon_new = fix_bug_001(polygon_new)
    #         polygon_new = fix_bug_006(polygon_new)
    #         polygon_new = fix_bug_003(polygon_new)
    #         polygon_new = remove_point_on_segment_0(polygon_new)
    #         polygon_new = fix_bug_001(polygon_new)
    #         polygon_new = fix_bug_002(polygon_new)
    #         polygon_new = remove_point_on_segment(polygon_new)
    #         list_polygon_new.append(polygon_new)
    #     except Exception:
    #         list_polygon_new.append(polygon) 
    # return list_polygon_new
    p_corr = Pool(processes=core_of_computer)
    result_list_polygon = p_corr.map(partial(correcting_pool), list_polygon)
    p_corr.close()
    p_corr.join()
    return result_list_polygon
  

def correcting_pool(polygon):
    try:
        polygon_new = fix_bug_001(polygon)
        polygon_new = fix_bug_001(polygon_new)
        polygon_new = fix_bug_006(polygon_new)
        polygon_new = fix_bug_003(polygon_new)
        polygon_new = remove_point_on_segment_0(polygon_new)
        polygon_new = fix_bug_001(polygon_new)
        polygon_new = fix_bug_002(polygon_new)
        polygon_new = fix_bug_007(polygon_new)
        polygon_new = remove_point_on_segment(polygon_new)
    except Exception:
        polygon_new = polygon
    return polygon_new

