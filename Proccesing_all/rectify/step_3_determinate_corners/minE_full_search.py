import numpy as np
from multiprocessing.pool import Pool
from functools import partial
import multiprocessing
core_of_computer = multiprocessing.cpu_count()
# import os
# path = os.getcwd()
# parent_path = os.sep.join(path.split(os.sep)[:-1])
# import sys
# sys.path.insert(0, parent_path)
from lib.geometry_calculating import ee,find_point_index,find_contour_rdp,find_max_angle,index_angle_to_first


def V_r(n,m,V_error):
    if n<2:
        return V_error[n]
    else:
        for j in range(1,n):
            error = ee(j-1,n-1,m)
            V_error[n,j] = error
        return V_error[n]
def dict_error(cnt,N):
    cnt1 = np.asarray(cnt, dtype=np.float32)
    core = multiprocessing.cpu_count()
    V_error=np.zeros((N+1,N+1))
    p = Pool(processes=core_of_computer)
    nums = []
    for n in range(0,N+1):
        nums.append(n)
    result = p.map(partial(V_r, m=cnt1,V_error=V_error), nums)
    p.close()
    p.join()
    V_error = np.asarray(result,dtype=np.float32)
    return V_error

def Search(N,M,V_error):
    D = np.zeros((N+1,N+1))
    A = np.zeros((N+1,N+1),dtype=int)
    H = np.zeros(M+1,dtype=int)
    D[1,0] = 0
    for n in range(2,N+1):
        D[n,0] = float("+inf")

    for m in range(1,M+1):
        for n in range(m,N+1):
            d_min = float("+inf")
            j_min = 0
            for j in range(m,n):
                d = D[j,m-1] + V_error[n,j]
                if d < d_min:
                    d_min = d
                    j_min = j
            D[n,m]=d_min
            A[n,m]=j_min
    H[M]=N
    for m in range(M,1,-1):
        H[m-1]= A[H[m],m]
    e = D[N,M]
    # print(H)
    return e,H


def full_search(contours2):
    cntstest = []
    for cnt in contours2:
        cnt_rdp,M = find_contour_rdp(cnt, 2.5)
    #        print(M)
        i_max = find_max_angle(cnt_rdp)
        point_optimis = cnt_rdp[i_max]#find optimis point use rdp
        point_op_index = find_point_index(point_optimis,cnt) #find optimis point index in cnt
        cnt = index_angle_to_first(cnt,point_op_index)
        listcnt = cnt.tolist()
        listcnt.append(listcnt[0])     #first point to first of cnt
        cnt = np.array(listcnt, dtype=np.int32)
        N = len(cnt)
        cnt_rdp1,M = find_contour_rdp(cnt, 2.5)# new cnt_rdp with new contours
        if M<60 and N<800:        #if rpd >=60 time cal long. iam use rdp for it
            V_error = dict_error(cnt,N)
            Ers,K_list = Search(N,M,V_error)
            # print(K_list)
            # print(N,M)
            # cntx,M = find_contour_rdp(cnt, 2.5)
            cntt = []
            for i in range(1,M+1):
                cntt.append(cnt[K_list[i]-1])
            cntx = np.asarray(cntt, dtype=np.int32)
            cntstest.append(cntx)
        else:
            cntstest.append(cnt_rdp1)
    return cntstest
    