import numpy as np
from rdp import rdp
from shapely.geometry import Point
import math

def area_triangle(p1,p2,p3):
    a=calculate_distance(p1,p2)
    b=calculate_distance(p2,p3)
    c=calculate_distance(p3,p1)

    s= (a+b+c)/2
    area=(s*(s-a)*(s-b)*(s-c))**0.5
    return area

    #tim giao diem giua hai vecto 1line la 2 point tao thanh chi phuong
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y
def projection_point_on_line(p1, p2, q):
    k = ((p2[1]-p1[1])*(q[0]-p1[0])-(p2[0]-p1[0])*(q[1]-p1[1]))/((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)
    hx = q[0] - k * (p2[1]-p1[1])
    hy = q[1] + k * (p2[0]-p1[0])
    H=np.asarray([hx,hy])
    return H


    #tinh goc giua hai vecto
def angle_vecto(v1,v2):
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    if cosine_angle < -1.0:
        cosine_angle = -1.0
    if cosine_angle > 1.0:
        cosine_angle = 1.0
    angle = np.arccos(cosine_angle)
    inner_angle = np.degrees(angle)
    return inner_angle

    #khoang cach giua hai diem
def calculate_distance(p1,p2):
    d=math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    return d

def find_contour_rdp(cnt,epsilon):
    #@author: AnhNT
    list_point = rdp(cnt,epsilon)
    return list_point,len(list_point)

def ee(j,n,cnt): # error of two point in polygon
    x = []

    x1,y1 = cnt[j][0]
    x2,y2 = cnt[n][0]
    a = (y2-y1)
    b = -(x2-x1)
    c = (x2*y1)-(y2*x1)
    e = math.sqrt(a**2+b**2)
    if e == 0.0:
        rs = 0.0
    else:
        for i in range(j+1,n):
            x0,y0 = cnt[i][0]
    #        d = abs((y2-y1)*x0 -(x2-x1)*y0+(x2*y1)-(y2*x1))/math.sqrt((y2-y1)**2+(x2-x1)**2)
    #        d = abs(a*x0 +b*y0 + c)/math.sqrt(a**2+b**2)
            d = abs(a*x0 +b*y0 + c)/e
    #        p3 = cnt[i][0]
    #        d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
            x.append(d)
        rs = sum(x)
    return rs

def linear_approximation(arr):
    M=[]#tap nay tra ve he so cac doan xap xi tuyen tinh
    x=[]#tap nay la tap cac hoanh do
    y=[]#tap nay la tap cac tung do tuong ung
    for i in range(len(arr)):
        x.append(arr[i][0])
        y.append(arr[i][1])
    n=x.count(x[0])
    l=len(x)
    x=np.array(x)
    y=np.array(y)
    if n<l:
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y)[0]
        M.append(-m)
        M.append(1)
        M.append(c)
    if n==l:
        M.append(1)
        M.append(0)
        M.append(x[0])
    return M

    #sxtt tra ve cai canh

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    result = np.asarray([x,y])
    return result

def index_angle_to_first(cnt,point_index):# index to fist in contours
    newcnt = []
    listcnt = list(cnt)
    for i in range(point_index,len(cnt)):
        newcnt.append(listcnt[i])
    for j in range(0,point_index):
        newcnt.append(listcnt[j])
    cnt_rs = np.asarray(newcnt, dtype=np.int32)
    return cnt_rs

def find_point_index(point,cnt):
    list_point = cnt.tolist()
    point_in_list = point.tolist()
    point_index = list_point.index(point_in_list)
    return point_index

def find_max_angle(cnt):
    d_max = 0
    i_max = 0
    for i in range(len(cnt)-1):
        d_point_i = Point(cnt[i][0]).distance(Point(cnt[i+1][0]))
        if d_point_i > d_max:
            d_max = d_point_i
            i_max = i
    d_point_i = Point(cnt[len(cnt)-1][0]).distance(Point(cnt[0][0]))
    if d_point_i > d_max:
        d_max = d_point_i
        i_max = i
    return i_max

def remove_duple_item(cnt):
    index=[]
    for i in range(len(cnt)-1):
        x1=cnt[i][0]
        y1=cnt[i][1]
        x2=cnt[i+1][0]
        y2=cnt[i+1][1]
        if x1 == x2 and y1==y2:
            index.append(i)
        if x1 != x2 or y1 != y2:
            continue
    for index in sorted(index, reverse=True):
        del cnt[index]
    if cnt[0][0]==cnt[len(cnt)-1][0] and cnt[0][1]==cnt[len(cnt)-1][1]:
        del cnt[len(cnt)-1]
    return cnt

def sort_array_base_on_index(arr,index):
    A=[]
    for i in range(index,len(arr)):
        A.append(arr[i])
    for i in range(index):
        A.append(arr[i])
    return A

def vector_of_degrees(alpha):
    x = np.cos(np.deg2rad(alpha))
    y = np.sin(np.deg2rad(alpha))
    vector_rs = np.array([x,y])
    return vector_rs


