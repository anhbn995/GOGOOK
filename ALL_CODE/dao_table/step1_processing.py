
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
from PIL import ImageEnhance
from PIL import Image
import argparse
from PIL import Image
from tqdm import *
import glob,os

'''
    - input :đường dẫn file pdf 
    - input2 : thu muc out anh crop
    - out: list duong dan den anh da dc crop
'''

# ham lay list anh jpg
'''
    - input 
        - input_flie_path : đường dẫn file pdf
        - out_dir : thư mục lưu anh crop

'''
def get_path_jpg(input_file_path,out_dir):
    list_path_jpg = []

    images = convert_from_path(input_file_path,dpi=280)
    out = out_dir
    # os.makedirs(out)
    for i, image in enumerate(images):
        out_file = out +f"/{i}.PNG"
        # print(out_file)
        image = np.array(image)
        image = get_square_box_from_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image.astype(np.uint8)
        
        img = Image.fromarray(image)
        img.save(out_file, "PNG")
        list_path_jpg.append(out_file)
    
    return list_path_jpg



def preprocess(img, factor: int):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(img).enhance(factor)
    if gray.std() < 30:
        enhancer = ImageEnhance.Contrast(enhancer).enhance(factor)
    return np.array(enhancer)

def group_h_lines(h_lines, thin_thresh):
    new_h_lines = []
    while len(h_lines) > 0:
        thresh = sorted(h_lines, key=lambda x: x[0][1])[0][0]
  
        lines = [line for line in h_lines if thresh[1] -
                 thin_thresh <= line[0][1] <= thresh[1] + thin_thresh]
        h_lines = [line for line in h_lines if thresh[1] - thin_thresh >
                   line[0][1] or line[0][1] > thresh[1] + thin_thresh]
    
        x = []
        for line in lines:
            x.append(line[0][0])
            x.append(line[0][2])
   
        x_min, x_max = min(x) - int(10*thin_thresh), max(x) + int(10*thin_thresh)
  
     
        new_h_lines.append([x_min, thresh[1], x_max, thresh[1]])
    return new_h_lines

def group_v_lines(v_lines, thin_thresh):
    new_v_lines = []
    while len(v_lines) > 0:
        thresh = sorted(v_lines, key=lambda x: x[0][0])[0][0]
        lines = [line for line in v_lines if thresh[0] -
                 thin_thresh <= line[0][0] <= thresh[0] + thin_thresh]
        v_lines = [line for line in v_lines if thresh[0] - thin_thresh >
                   line[0][0] or line[0][0] > thresh[0] + thin_thresh]
        y = []
        for line in lines:
            y.append(line[0][1])
            y.append(line[0][3])
        y_min, y_max = min(y) - int(4*thin_thresh), max(y) + int(4*thin_thresh)
        new_v_lines.append([thresh[0], y_min, thresh[0], y_max])
    return new_v_lines

def seg_intersect(line1: list, line2: list):
    a1, a2 = line1
    b1, b2 = line2
    da = a2-a1
    db = b2-b1
    dp = a1-b1

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1

def get_bottom_right(right_points, bottom_points, points):

    for right in right_points:
       
        for bottom in bottom_points:
            
            if [right[0], bottom[1]] in points:
             
                return right[0], bottom[1]
       
    return None, None

def euclidian_distance(point1, point2):
    # Calcuates the euclidian distance between the point1 and point2
    #used to calculate the length of the four sides of the square 
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

def order_corner_points(corners):
    # The points obtained from contours may not be in order because of the skewness  of the image, or
    # because of the camera angle. This function returns a list of corners in the right order 
    sort_corners = [(corner[0][0], corner[0][1]) for corner in corners]
    sort_corners = [list(ele) for ele in sort_corners]
    x, y = [], []

    for i in range(len(sort_corners[:])):
        x.append(sort_corners[i][0])
        y.append(sort_corners[i][1])

    centroid = [sum(x) / len(x), sum(y) / len(y)]

    for _, item in enumerate(sort_corners):
        if item[0] < centroid[0]:
            if item[1] < centroid[1]:
                top_left = item
            else:
                bottom_left = item
        elif item[0] > centroid[0]:
            if item[1] < centroid[1]:
                top_right = item
            else:
                bottom_right = item

    ordered_corners = [top_left, top_right, bottom_right, bottom_left]

    return np.array(ordered_corners, dtype="float32")

def image_preprocessing(image, corners):
    # This function undertakes all the preprocessing of the image and return  
    ordered_corners = order_corner_points(corners)
    # print("ordered corners: ", ordered_corners[0])
    top_left, top_right, bottom_right, bottom_left = ordered_corners
    top_left = top_left+np.array([-35.,-35.])
    top_right = top_right+np.array([35.,-35.])
    bottom_right = bottom_right+np.array([35.,35.])
    bottom_left = bottom_left+np.array([-35.,35.])
    ordered_corners[0] = top_left
    ordered_corners[1] = top_right
    ordered_corners[2] = bottom_right
    ordered_corners[3] = bottom_left
    # print("ordered corners 2:",ordered_corners)
    # Determine the widths and heights  ( Top and bottom ) of the image and find the max of them for transform 

    width1 = euclidian_distance(bottom_right, bottom_left)
    width2 = euclidian_distance(top_right, top_left)

    height1 = euclidian_distance(top_right, bottom_right)
    height2 = euclidian_distance(top_left, bottom_left)

    width = max(int(width1), int(width2)) 
    height = max(int(height1), int(height2)) 
    

    # To find the matrix for warp perspective function we need dimensions and matrix parameters
    dimensions = np.array([[0, 0], [width, 0], [width, height],
                           [0, height]], dtype="float32")

    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    transformed_image = cv2.warpPerspective(image, matrix, (width, height))

    #Now, chances are, you may want to return your image into a specific size. If not, you may ignore the following line
    # transformed_image = cv2.resize(transformed_image, (252, 252), interpolation=cv2.INTER_AREA)

    return transformed_image ,ordered_corners




def seg_intersect(line1: list, line2: list):
    a1, a2 = line1
    b1, b2 = line2
    da = a2-a1
    db = b2-b1
    dp = a1-b1

    def perp(a):
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1


def get_bottom_right(right_points, bottom_points, points):
    for right in right_points:
        for bottom in bottom_points:
            if [right[0], bottom[1]] in points:
                return right[0], bottom[1]
    return None, None

def get_square_box_from_image(image):
    # This function returns the top-down view of the puzzle in grayscale.
    # 

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    adaptive_threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    corners = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = corners[0] if len(corners) == 2 else corners[1]
    corners = sorted(corners, key=cv2.contourArea, reverse=True)
    for corner in corners:
        length = cv2.arcLength(corner, True)
        approx = cv2.approxPolyDP(corner, 0.015 * length, True)
        # print(approx)
      

        puzzle_image,box= image_preprocessing(image, approx)
        break
    return puzzle_image


if __name__ == '__main__':
    # input_file_pdf = '/home/skymap/data/CHUYENDOISOVT/NEW_DATA_TRAIN_OBJ/outdata.pdf'
    out_dir_crop = '/home/skymap/data/CHUYENDOISOVT/OUTDATA/anh_chua_crop/crop'
    
    input_pdf_dir = '/home/skymap/data/CHUYENDOISOVT/OUTDATA/anh_chua_crop'

    # list_path = get_path_jpg(input_file_pdf,out_dir_crop)
    # print(len(list_path))
    # print(list_path)
    for fp_img in tqdm(glob.glob(os.path.join(input_pdf_dir,'*.pdf'))):
        name = os.path.basename(fp_img)
        name1 = name.split()
        name2 = name1[0].split('.')[0]

        list_path = get_path_jpg(fp_img,out_dir_crop,name2)

