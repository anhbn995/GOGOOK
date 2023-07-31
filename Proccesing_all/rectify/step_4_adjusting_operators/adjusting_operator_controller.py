import time
import cv2
import numpy as np
from rdp import rdp
from multiprocessing.pool import Pool
from functools import partial
# import cv2
import matplotlib.patches as patches
import gdal, gdalconst, osr, ogr
import matplotlib.pyplot as plt
#from PIL import Image
from shapely.geometry import Point, LineString, Polygon
#from math import pi
from pyproj import Proj, transform
from math import pi
import math
import multiprocessing
from numpy.linalg import norm
from step_4_adjusting_operators.divide_trust_untrust_contours import threshold_contour
from step_4_adjusting_operators.fix_trust_contours import strange_contour_trust_main
from step_4_adjusting_operators.fix_untrust_contours import fix_untrust_contours_by_bounding_rect,fix_untrust_unbound
core_of_computer = multiprocessing.cpu_count()
T_SL = 0.3
T_Projection_Final = 2/0.3
T_Deviation = 2/0.3
T_Ratio = 0.1
T_Footprint = 0.85

T_area=0.2
T_distance=20
T_distance_corner=1
T_length=3
T_area_intersection=0.5


def adjusting_operators_main(list_cntstest,list_translative):
    print("begin adjusting_operators_main")
    list_trust_contours, list_untrust_contours,list_trust_vector,list_untrust_vectors,list_trust_translative,list_untrust_translative = threshold_contour(list_cntstest,T_SL,list_translative)
    list_fixed_trust_contours = strange_contour_trust_main(list_trust_contours,list_trust_vector) 
    list_fixed_untrust_contours,list_untrust_unbound,list_untrust_unbound_vector,list_bound_untrust_translative, list_unbound_untrust_translative = fix_untrust_contours_by_bounding_rect(list_untrust_contours,list_untrust_vectors,list_untrust_translative)
    list_fixed_untrust_unbound = fix_untrust_unbound(list_untrust_unbound,list_untrust_unbound_vector)
    list_translative_new = list(list_trust_translative)
    all_fixed_contour = list(list_fixed_trust_contours)
    all_fixed_contour.extend(list_fixed_untrust_contours)
    list_translative_new.extend(list_bound_untrust_translative)
    all_fixed_contour.extend(list_fixed_untrust_unbound)
    list_translative_new.extend(list_unbound_untrust_translative)
    print("adjusting_operators_main")

    return all_fixed_contour,list_translative_new
