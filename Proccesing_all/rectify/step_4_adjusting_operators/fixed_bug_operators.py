# This Python file uses the following encoding: utf-8

import numpy as np
from lib.geometry_calculating import angle_vecto,line_intersection, calculate_distance,area_triangle
from numpy.linalg import norm
from lib.convert_datatype import contour_to_list_array,list_array_to_contour, list_contour_to_list_list_array, list_list_array_to_list_contour, convert_numpy_to_list

# Định nghĩa các thao tác nẳn chỉnh để xử lý các bug
# Đặt tên fix tương ứng với bug. 
# Ví dụ: bug001 => fix001 

# Hàm main fix tất cả các lỗi
# Đầu vào:
# Đầu ra:
def fix_all_main(list_contours,list_cntstest,T_area,T_distance,T_distance_corner,T_length,T_area_intersection):
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
        point_after_linear_approximation=find_corner(parameter_and_point[0],parameter_and_point[1],T_distance_corner)
        #tim ra truc va cho no len dau
        point_axis_first=sort_distance_and_find_axis(point_after_linear_approximation) 
        result_corner=fix010(point_axis_first,T_area,T_distance,T_length,T_area_intersection)
        list_building.append(result_corner)

    list_building_rs = list_list_array_to_list_contour(list_building)
    return list_building_rs

# ==================================================================
# ====================== Các fixed bugs ===========================
# ==================================================================
# Danh sách các fix: 002, 003, 006, 007, 008, 009, 010
    

# Trường hợp: Hai cạnh cắt nhau tạo thành tam giác
# Tham số: T_area_intersection (diện tích phần giao)
# Phương pháp xử lý: 
# Name: remove_two_segment_intersection
def fix002(numpy_list,T_area_intersection):
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[0],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[1],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[2],0)

    for i in range(len(numpy_list)-3):
        p1=numpy_list[i]
        p2=numpy_list[i+1]
        p3=numpy_list[i+2]
        p4=numpy_list[i+3]

        G = line_intersection((p1,p2),(p3,p4))
        if G != False:
            d1=calculate_distance(p1,G)
            d2=calculate_distance(p2,G)
            d3=calculate_distance(p3,G)
            d4=calculate_distance(p4,G)
            d=calculate_distance(p1,p2)
            eps1=(d1+d2)/d
            eps2=(d3+d4)/d
            if (0.99 < eps1 < 1.01) and (0.99 < eps2 < 1.01):# and (d1 < d) and (d3 < d) and (d4 < d) and (d2 < d):
                area_small = area_triangle(p2,p3,G)
                area_big = area_triangle(p1,p4,G)
                epsilon = area_small/area_big
                if epsilon < T_area_intersection:
                    I=(G+p1)/2
                    numpy_list=np.delete(numpy_list,i+1,0)
                    numpy_list=np.delete(numpy_list,i+1,0)
                    numpy_list=np.insert(numpy_list,i+1,G,0)
                    numpy_list=np.insert(numpy_list,i+1,I,0)
        else:
            continue
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
#    print("gasdhadgasdasgdashasdhasd",numpy_list)
    return numpy_list


# Trường hợp: đoạn ở giữa rất ngắn hơn so với 2 đoạn lân cận
# Tham số: tỷ lệ T_length
# Giải pháp: Merge 4 điểm thành 2 điểm 
# Name: merge_four_points
def fix003(numpy_list,T_length):
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[0],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[1],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[2],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[3],0)
#    print(numpy_list)
    M=numpy_list[len(numpy_list)-1]
    N=numpy_list[len(numpy_list)-2]
    vec_axis=numpy_list[1]-numpy_list[0]
    A=(M+N)/2
    B=A+vec_axis
    for i in range(1,len(numpy_list)-4):
        p0=numpy_list[i-1]
        p1=numpy_list[i]
        p2=numpy_list[i+1]
        p3=numpy_list[i+2]
        p4=numpy_list[i+3]
        p5=numpy_list[i+4]

        vec1=p2-p1
        vec3=p4-p3

        J=projection_point_on_line(p3,p4,p2)
        distance_p2_to_p3p4=calculate_distance(p2,J)

        if distance_p2_to_p3p4 < T_length:
            angle1_axis=angle_vecto(vec1,vec_axis)
            angle2_axis=angle_vecto(vec3,vec_axis)
            if (angle1_axis < 1 and angle2_axis < 1) or (angle1_axis > 179 and angle2_axis > 179):
                H=projection_point_on_line(A,B,p1)
                K=projection_point_on_line(A,B,p4)
                l1=calculate_distance(H,p1)
                l2=calculate_distance(K,p4)
                if l1 > l2:
                    I = line_intersection((p1,p2),(p4,p5))
                    m=calculate_distance(I,p4)
                    if m<10:
                        numpy_list=np.delete(numpy_list,i+2,0)
                        numpy_list=np.insert(numpy_list,i+2,I,0)
                    else:
                        continue
                if l1 < l2:
                    I = line_intersection((p3,p4),(p1,p0))
                    m=calculate_distance(I,p1)
                    if m<10:
                        numpy_list=np.delete(numpy_list,i+1,0)
                        numpy_list=np.insert(numpy_list,i+1,I,0)
                    else:
                        continue
            if 89 < angle1_axis <91 and 89 < angle2_axis <91:
                u = np.array([-vec_axis[1],vec_axis[0]])
#                M=(A+B)/2
                I= A+u
                H=projection_point_on_line(A,I,p1)
                K=projection_point_on_line(A,I,p4)
                l1=calculate_distance(H,p1)
                l2=calculate_distance(K,p4)
                if l1 > l2:
                    I = line_intersection((p1,p2),(p4,p5))
                    m=calculate_distance(I,p4)
                    if m < 10:
                        numpy_list=np.delete(numpy_list,i+2,0)
                        numpy_list=np.insert(numpy_list,i+2,I,0)
                if l1 < l2:
                    I = line_intersection((p3,p4),(p1,p0))
                    m=calculate_distance(I,p1)
                    if m < 10:
                        numpy_list=np.delete(numpy_list,i+1,0)
                        numpy_list=np.insert(numpy_list,i+1,I,0)
                    else:
                        continue
        else:
            continue
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=fix007(numpy_list) #remove_point_on_segment
    return numpy_list
    
 # Trường hợp: tồn tại 3 điểm thằng hàng theo thứ tự xuất hiện điểm 
 # là p1,p2,p3 thì p2 nằm trên đoạn p1,p3  
 # Tham số: None
 # Giải pháp: xóa p2
 # Name: remove_remove_point_on_segment_0
def fix006(numpy_list):
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[0],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[1],0)
    list_numpy_new=[]
    for i in range(1,len(numpy_list)-1):
        p1=numpy_list[i-1]
        p2=numpy_list[i]
        p3=numpy_list[i+1]

        #cac vecto lien tiep
        vec1=p2-p1
        vec2=p3-p2

        angle=angle_vecto(vec1,vec2)

        if angle == 0:
            continue
        else:
            list_numpy_new.append(numpy_list[i])
    #muc dich lai cho truc len dau
    list_numpy_new.insert(0,list_numpy_new[len(list_numpy_new)-1])
    list_numpy_new=remove_duple_item(list_numpy_new)
    return list_numpy_new

 # Trường hợp: tồn tại 3 điểm thằng hàng theo thứ tự xuất hiện điểm 
 # p2 không nằm trên đoạn p1,p3 mà nằm ngoài đoạn 
 # Tham số: None
 # Giải pháp: xóa p2
 # Name: remove_point_on_segment
def fix007(numpy_list):#p1,p2,p3 la 3 diem lien tiep
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[0],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[1],0)

    list_numpy_new=[]
    for i in range(1,len(numpy_list)-1):
        p1=numpy_list[i-1]
        p2=numpy_list[i]
        p3=numpy_list[i+1]

        #cac vecto lien tiep
        vec1=p2-p1
        vec2=p3-p2

        angle=angle_vecto(vec1,vec2)

        if angle == 180:
            continue
        else:
            list_numpy_new.append(numpy_list[i])

    list_numpy_new.insert(0,list_numpy_new[len(list_numpy_new)-1])
    list_numpy_new=remove_duple_item(list_numpy_new)
    list_numpy_new=fix006(list_numpy_new)
    return list_numpy_new
    
 # Trường hợp: 4 điểm liên tiếp p1,p2,p3,p4 và khi xét góc nào mà
 # bị vát và cần nắn vuông
 # Tham số: tỷ lệ diện tích T_area
 # Giải pháp: gộp 4 điểm thành 3 điểm 
 # Name: perpendicular_angel_45
def fix008(point_sort,T_area):
    point_sort=fix007(point_sort)
    point_sort.extend((point_sort[0],point_sort[1],point_sort[2],point_sort[3]))
    A=point_sort[0]
    B=point_sort[1]
    u=B-A
    index=[len(point_sort)-2,len(point_sort)-1,len(point_sort)-3]
    for i in range(len(point_sort)-3):
        p1=point_sort[i]
        p2=point_sort[i+1]
        p3=point_sort[i+2]
        p4=point_sort[i+3]

        vec1=p2-p1
        vec2=p3-p2
        vec3=p4-p3

        angle=angle_vecto(vec1,u)
        angle1=angle_vecto(vec1,vec2)
        angle2=angle_vecto(vec2,vec3)
        angle3=angle_vecto(vec1,vec3)

        if (0 < angle1 < 80) and (80 < angle3 < 110) and (0 < angle2 < 80):
            if (angle < 10) or (angle > 170) or (80 < angle < 110):
                I=line_intersection((p1,p2),(p3,p4))
                area_small=area_triangle(p2,p3,I)
                area_big=area_triangle(p1,p4,I)
                epsilon=area_small/area_big

                if epsilon < T_area:
                    point_sort=np.delete(point_sort,i+1,0)
                    point_sort=np.insert(point_sort,i+1,I,0)
                    index.append(i+2)
    point_sort=np.delete(point_sort,index,0)
    point_sort=fix007(point_sort)
    return point_sort

   
  
 # Trường hợp: những đoạn thẳng gần vuông góc với nhau
 # 1 cạnh song song hoặc vuông góc với trục
 # Tham số: None
 # Giải pháp: nắn cho chúng vuông góc với nhau
 # Name: square_shaped
def fix009(numpy_list):

    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[0],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[1],0)
    numpy_list=np.insert(numpy_list,len(numpy_list),numpy_list[2],0)

    A=numpy_list[0]
    B=numpy_list[1]
    vec_axis=B-A

    for i in range(len(numpy_list)-3):
        p1=numpy_list[i]
        p2=numpy_list[i+1]
        p3=numpy_list[i+2]
        p4=numpy_list[i+3]

        vec1 = p2-p1
        vec2 = p3-p2
        vec3 = p4-p3

        angle1=angle_vecto(vec1,vec2)
        angle2=angle_vecto(vec1,vec3)
        angle=angle_vecto(vec1,vec_axis)

        if (0 < angle1 < 88 and angle2 < 3) or (angle1 > 92 and angle2 < 3):
            if 88 < angle < 92:
                u=np.array([-vec_axis[1],vec_axis[0]])
                C=A+u
                I = (p2+p3)/2
                J=projection_point_on_line(A,C,I)
                H=line_intersection((I,J),(p1,p2))
                K=line_intersection((I,J),(p3,p4))
                numpy_list = np.delete(numpy_list,i+1,0)
                numpy_list = np.insert(numpy_list,i+1,H,0)
                numpy_list = np.delete(numpy_list,i+2,0)
                numpy_list = np.insert(numpy_list,i+2,K,0)
            if angle < 2 or angle > 178:
                I = (p2+p3)/2
                J=projection_point_on_line(A,B,I)

                H=line_intersection((I,J),(p1,p2))
                K=line_intersection((I,J),(p3,p4))
                numpy_list = np.delete(numpy_list,i+1,0)
                numpy_list = np.insert(numpy_list,i+1,H,0)

                numpy_list = np.delete(numpy_list,i+2,0)
                numpy_list = np.insert(numpy_list,i+2,K,0)

    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    numpy_list=np.delete(numpy_list,len(numpy_list)-1,0)
    return numpy_list

 # Trường hợp: những đoạn thẳng gần song song hoặc gần vuông góc với trục
 # Tham số: T_length la he so khi nan cai song song co doan o giua nho
 # Giải pháp: nắn thẳng
 # Name: straight
def fix010(point_sort,T_area,T_distance,T_length,T_area_intersection):
    point= fix008(point_sort,T_area) #perpendicular_angel_45 
    point=np.insert(point,len(point),point[0],0)
    point=np.insert(point,len(point),point[1],0)

#    print("cai nay",point)ix010
    p=point[0]
    q=point[1]

    vecto_axis=q-p
    for i in range(1,len(point)-2):
        p1=point[i-1]
        p2=point[i]
        p3=point[i+1]
        p4=point[i+2]
        vec=p3-p2
        angle=angle_vecto(vec,vecto_axis)

        if 70 < angle < 110:
            I=(p3+p2)/2
            H=projection_point_on_line(p,q,I)
            K=line_intersection((H,I),(p1,p2))
            if K!=False:
                l1=calculate_distance(K,p2)
                if l1 < T_distance:
                    point=np.delete(point,i,0)
                    point=np.insert(point,i,K,0)
            if K == False:
                    continue

            M=line_intersection((H,I),(p3,p4))
            if M!=False:
                l2=calculate_distance(M,p3)
                if l2 < T_distance:
                    point=np.delete(point,i+1,0)
                    point=np.insert(point,i+1,M,0)
            if M == False:
                continue

        if angle < 20 or angle > 160:
            I=(p3+p2)/2
            H=I+vecto_axis
            K=line_intersection((H,I),(p1,p2))
            if K!=False:
                l1=calculate_distance(K,p2)
                if l1 < T_distance:
                    point=np.delete(point,i,0)
                    point=np.insert(point,i,K,0)
            if K == False:
                continue
            M=line_intersection((H,I),(p3,p4))
            if M!=False:
                l2=calculate_distance(M,p3)
                if l2 < T_distance:
                    point=np.delete(point,i+1,0)
                    point=np.insert(point,i+1,M,0)
            if M == False:
                continue
    point=np.delete(point,(len(point)-1),0)
    point=fix007(point) #remove_point_on_segment
    point=list(point)
    point=fix008(point,T_area) #perpendicular_angel_45
#
    point=fix002(point,T_area_intersection) #remove_two_segment_intersection
    point=fix003(point,T_length) #merge_four_points
    point=fix009(point) #square_shaped
    point=remove_duple_item(point)
    return point
# ==================================================================
# ====================== Các hàm phụ trợ ===========================
# ==================================================================
    
    #xoa ca phan tu trung lien tiep trong mang cac list array diem dau khong trung diem cuoi
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

    #xác định hình chiếu của q lên duowng thẳng p1p2
def projection_point_on_line(p1, p2, q):
    k = ((p2[1]-p1[1])*(q[0]-p1[0])-(p2[0]-p1[0])*(q[1]-p1[1]))/((p2[1]-p1[1])**2+(p2[0]-p1[0])**2)
    hx = q[0] - k * (p2[1]-p1[1])
    hy = q[1] + k * (p2[0]-p1[0])
    H=[hx,hy]
    return H

    #sap xep lai mang theo index nao do dung dau cac index truoc no se lai duoc chuyen xuong cuoi
def sort_array_base_on_index(arr,index):
    A=[]
    for i in range(index,len(arr)):
        A.append(arr[i])
    for i in range(index):
        A.append(arr[i])
    return A


    #tim canh dau tien
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

    #tim cac doan co the nan thang
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

 #sxtt tra ve cai canh ??????????????????
def find_edge(arr1,arr2):
    arr1=convert_numpy_to_list(arr1)
    arr2=convert_numpy_to_list(arr2)

    for i in arr1:
        if i==arr2[0]:
            arr1=sort_array_base_on_index(arr1,arr1.index(i))
            break
    arr1.append(arr1[0])
    arr2.append(arr2[0])

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

#"---------------------tim truc tot nhat--------------------------------------"
    #dau kkhong trung cuoi dau nhe
def sort_distance_and_find_axis(point_linear_approximation):
    R=[]
    point_linear_approximation.append(point_linear_approximation[len(point_linear_approximation)-1])
    for i in range(len(point_linear_approximation)-1):
        point1=point_linear_approximation[i]
        point2=point_linear_approximation[i+1]
        y=calculate_distance(point1,point2)
        R.append(y)
    C = list(R)
    #C=R.copy()
    C.sort(reverse=True)


    index_decrease=sorted(range(len(R)),key=lambda k:R[k])
    index_decrease.reverse()
    point=[]
    for i in index_decrease:
        x=[point_linear_approximation[i],point_linear_approximation[i+1]]
        point.append(x)

    C=np.array(C)
    average_length=sum(C)/len(C)
    M=np.where(C >= average_length)
    len_M=len(M[0])
    number_edge = []
    for i in range(len_M):
        u=point[i][1]-point[i][0]
        u=np.array(u)
        temp=0
        for j in range(len(point_linear_approximation)-1):
            point1=point_linear_approximation[j]
            point2=point_linear_approximation[j+1]
            vec=point2-point1
            vec=np.array(vec)
            angle=angle_vecto(u,vec)
            if 88 < angle < 92 or angle < 2 or angle > 178:
                temp=temp+1
        number_edge.append(temp)
    index=number_edge.index(max(number_edge))

    t=point[index][0]
    p=t.tolist()

    point_linear_approximation=np.array(point_linear_approximation)
    q=point_linear_approximation.tolist()

    for i in range(len(q)):
        if q[i]==p:
            result=sort_array_base_on_index(point_linear_approximation,i)
    result=remove_duple_item(result)
#    result.pop(len(result)-1)
    result=remove_duple_item(result)
    return result
