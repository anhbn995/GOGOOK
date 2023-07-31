import numpy as np
from lib.geometry_calculating import sort_array_base_on_index,remove_duple_item,angle_vecto,linear_approximation,calculate_distance
from lib.convert_datatype import list_list_array_to_list_contour,list_contour_to_list_list_array,convert_numpy_to_list

def find_edge(arr1,arr2):
    arr1=convert_numpy_to_list(arr1)
    arr2=convert_numpy_to_list(arr2)
    
    for i in arr1:
        if i==arr2[0]:
            arr1=sort_array_base_on_index(arr1,arr1.index(i))
            break
    arr1.append(arr1[0])
    arr2.append(arr2[0])
    # print(arr1)
    # print(arr2)
    k=[]
    m=[]
    for i in range(len(arr2)-1):
        start=arr1.index(arr2[i])
        end=arr1.index(arr2[i+1])+1
        if i == len(arr2)-2:
            end=len(arr1)
        z=[]
        for j in range(start,end):
            z.append(arr1[j])
            lenz=len(z)
            t=end-start
            if lenz==t:
                temp=linear_approximation(z)
                k.append(temp)
                m.append(z)
    return k,m

    #co canh tim goc tra ve dau khong trung cuoi
def find_corner(k,m,T_distance_coner):
    corner=[]
    k.append(k[0])
    m.append(m[0])
    for i in range(len(k)-1):
        a=np.array([[k[i][0],k[i][1]],[k[i+1][0],k[i+1][1]]])
        b=np.array([k[i][2],k[i+1][2]])
        point=np.linalg.solve(a,b)

        q=m[i][len(m[i])-1]
        q=np.array(q)
        p=m[i+1][0]
        p=np.array(p)
        l=calculate_distance(point,p)
        if l <= T_distance_coner:
            corner.append(point)
        else:
            corner.extend((p,q))
    corner=remove_duple_item(corner)
    return corner

def find_edge_first(arr):
    for i in range(len(arr)-2):
        p1=arr[i]
        p2=arr[i+1]
        p3=arr[i+2]

        v1=p2-p1
        v2=p3-p2

        angle=angle_vecto(v1,v2)
        #neu goc ma vuong thi se cho cai goc do len dau mang moi
        if 67.5 < angle < 112.5:
            k = i+1
            arr=sort_array_base_on_index(arr,k)
            break
    return arr


def find_segment_can_straight(arr):
    arr=find_edge_first(arr)
    arr=np.insert(arr,len(arr),arr[0],0)
    A=[arr[0]]
    for i in range(len(arr)-2):
        p1=arr[i]
        p2=arr[i+1]
        p3=arr[i+2]

        v1=p2-p1
        v2=p3-p2

        angle=angle_vecto(v1,v2)
        if 0 <= angle <= 20:
            continue
        if 20 < angle:
            A.append(p2)
    return A

def list_find_all_building_linear(list_contours,list_cntstest,T_distance_corner):
    list_building=[]
    contour_defau=list_contour_to_list_list_array(list_contours)
    contour_mine=list_contour_to_list_list_array(list_cntstest)
    for i in range(len(contour_defau)):
        con_defau=contour_defau[i]
        con_mine=contour_mine[i]

        segment_straight=find_segment_can_straight(con_mine)
        #he so va cac tap diem se tuyen tinh vaoi nhau
        parameter_and_point=find_edge(con_defau,segment_straight)
        #diem sau khi sxtt
        try:
            point_after_linear_approximation=find_corner(parameter_and_point[0],parameter_and_point[1],T_distance_corner)
        except Exception: 
            point_after_linear_approximation = con_mine
        
        #        print(point_after_linear_approximation)
        list_building.append(point_after_linear_approximation)
#    print("thang mat day nay",list_building[20])
    list_building_rs = list_list_array_to_list_contour(list_building)
    return list_building_rs
