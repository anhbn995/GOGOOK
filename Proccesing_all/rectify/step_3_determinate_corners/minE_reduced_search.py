import numpy as np
import multiprocessing
core_of_computer = multiprocessing.cpu_count()
# import os
# path = os.getcwd()
# parent_path = os.sep.join(path.split(os.sep)[:-1])
# import sys
# sys.path.insert(0, parent_path)
from lib.geometry_calculating import ee,find_point_index,find_contour_rdp,find_max_angle,index_angle_to_first




def Search(list_bottom_n,list_left_m,list_top_n,N,M,cnt):
#    v_error=v_error
    D = np.zeros((N+1,N+1))
    A = np.zeros((N+1,N+1),dtype=int)
    H = np.zeros(M+1,dtype=int)
    for n in range(2,N+1):
        D[n,0] = float("+inf")
        bot = list_bottom_n[n]-1
        left = list_left_m[bot]
        v_er = np.zeros(n)
        for j in range(left,n):
            v_er[n-j] = ee(j-1,n-1,cnt)
#        print(v_er)
#        print(list_bottom_n[n])
#        print(n)
#        print(list_top_n[n])
        for m in range(list_bottom_n[n],list_top_n[n]+1):
            d_min = float("+inf")
            j_min = 0
            for j in range(list_left_m[m-1],n):
                d = D[j,m-list_bottom_n[n]] + v_er[n-j]
#                print(d)
                if (d<d_min):
                    d_min = d
                    j_min = j
            D[n,m+1-list_bottom_n[n]]=d_min
            A[n,m+1-list_bottom_n[n]]=j_min
    E = D[N,M-list_bottom_n[N]]
    H[M]=N
    A=np.asarray(A)
#    print(A)
    D=np.asarray(D)
    for m in range(M,0,-1):
#        xx = H[m]
        H[m-1]= A[H[m],m+1-list_bottom_n[H[m]]]
    # print(E,H)
    return E,H


def cal_list_g_path(cnt_rdp,cnt):
    list_g_path = []
    for point in cnt_rdp:
        point_index = find_point_index(point,cnt)
        list_g_path.append(point_index+1)
    return list_g_path

def cal_list_left_m(M,list_g_path,c_1):
    left_m = []
    for m in range(M+1):
        if 0<=m<=c_1:
            left_m.append(m+1)
        elif c_1<m<=M:
            index = max(m+1,list_g_path[m-c_1])
            left_m.append(index)
    return left_m

def cal_list_right_m(N,M,c_2,list_g_path):
    right_m = []
    for m in range(M+1):
        if 0<=m<= M-c_2:
            index = min(N,list_g_path[m+c_2]-1)
            right_m.append(index)
        elif M-c_2+1<=m<=M:
            right_m.append(N)
    return right_m

def cal_list_bottom_n(list_right_m,N,M):
    bottom_n = []
    bottom_n.append(0)
    for i in range(list_right_m[0]):
        bottom_n.append(1)
    for m in range(1,M+1):
        for j in range(list_right_m[m-1]+1,list_right_m[m]+1):
            bottom_n.append(m)
    # print(len(bottom_n))
    return bottom_n
def cal_list_top_n(list_left_m,M,N):
    top_n = []
    for j in range(0,list_left_m[1]):
        top_n.append(1)
    for m in range(1,M):
        for j in range(list_left_m[m],list_left_m[m+1]):
            top_n.append(m)
    for k in range(list_left_m[M],N+1):
        top_n.append(M)
#    print(len(top_n))
    return top_n

def reduce_search(cnt):
#        listcnt = cnt.tolist()
#        listcnt.append(listcnt[0])
#        cnt = np.array(listcnt, dtype=np.int32)

    cnt_rdp,M = find_contour_rdp(cnt, 2.2)
    # print(M)
#        print(cnt)
#        print(cnt_rdp)
    i_max = find_max_angle(cnt_rdp)
    point_optimis = cnt_rdp[i_max]#find optimis point use rdp
    point_op_index = find_point_index(point_optimis,cnt) #find optimis point index in cnt
    cnt = index_angle_to_first(cnt,point_op_index)
#    listcnt = cnt.tolist()
#    listcnt.append(listcnt[0])
#    cnt = np.array(listcnt, dtype=np.int32)
    cnt_rdp,M = find_contour_rdp(cnt, 2.2)
    M = M
    N = len(cnt)
    if M<50 and N<600:
        W = 2*M
        c_1 = W//2
        c_2 = W-c_1
        list_g_path = cal_list_g_path(cnt_rdp,cnt)
        list_g_path.append(N+1)
        listcnt = cnt.tolist()
        listcnt.append(listcnt[0])
        cnt = np.array(listcnt)
        N = len(cnt)

        list_left_m = cal_list_left_m(M,list_g_path,c_1)
        list_right_m = cal_list_right_m(N,M,c_2,list_g_path)
        list_bottom_n = cal_list_bottom_n(list_right_m,N,M)
        list_top_n = cal_list_top_n(list_left_m,M,N)
        Ers,K_list = Search(list_bottom_n,list_left_m,list_top_n,N,M,cnt)
        cntt = []
        for i in range(len(K_list)-1):
            cntt.append(cnt[K_list[i]-1])
        cntx = np.asarray(cntt).astype(np.float32)
    else:
        cntx,M = find_contour_rdp(cnt, 2.5)
        cntx = cntx.astype(np.float32)
    return cntx
