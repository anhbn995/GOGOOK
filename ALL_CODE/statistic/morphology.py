import cv2
import numpy as np
from tqdm.notebook import tqdm


# choose kernel
def custom_kernel():
    """
        Tu vao sua tay ra hinh shape mong muon
    """
    kernel = np.ones((5,5),np.uint8)
    return kernel


def choose_kernel(size_kernel, shape_kernel):
    """
        shape: la hinh dang cua kernel, trong cac value sau ("elip", "rec", "cross", "custom")
        return gia tri kernel
    """

    if shape_kernel.upper() == "ELIP":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size_kernel,size_kernel))
    elif shape_kernel.upper() == "REC":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(size_kernel,size_kernel))
    elif shape_kernel.upper() == "CROSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(size_kernel,size_kernel))   
    elif shape_kernel.upper() == "CUSTOM":
        kernel = custom_kernel()
    else:
        print('shape bi sai: 1 trong 4 eg: "elip", "rec", "cross", "custom"')
    return kernel

# bat dau xu ly
def check_version_opencv_larger_4():
    version = cv2.__version__
    if float(version[:3]) >= 4:
        return True
    else:
        return False

def remove_area_small(mask, area_maximum, value_draw=255):
    # """
    #     Xoa nhung vung co kich thuoc < area bang gia tri mac dinh la 255

    #     INPUT:
    #         - mask: dang shape (w*h)
    #         - area: kich thuoc pixel
    #          - value_draw: gia tri dien vao
    #     OUTPUT:
    #         - Mask remove
    # """
    if check_version_opencv_larger_4():
        contours, _ = cv2.findContours(mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, _ = cv2.findContours(mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in tqdm(contours, desc='Run contour'):
        area = cv2.contourArea(contour)
        if area <= area_maximum:
            cv2.fillPoly(mask, [contour], value_draw)
    return mask


def dilation(mask, size_kernel=3, shape_kernel="elip", iterations=1):
    """
        Lam gian no hinh anh, to ra
        
        INPUT: 
            - mask: dang shape (w*h)
            - size_kernel: kich thuoc kernel. Default = 3
            - shape_kernel: 1 trong "elip", "rec", "cross", "custom". Default = elip
            - iterations: so lan lap, mac dinh 1
        OUTPUT:
            - Mask gian no to ra.
    """
    kernel = choose_kernel(size_kernel, shape_kernel)
    return cv2.dilate(mask, kernel, iterations = iterations)


def erosion(mask, size_kernel=3, shape_kernel="elip", iterations=1):
    """
        Lam co hinh anh, anh nho lai.
        
        INPUT: 
            - mask: dang shape (w*h)
            - size_kernel: kich thuoc kernel. Default = 3
            - shape_kernel: 1 trong "elip", "rec", "cross", "custom". Default = elip
            - iterations: so lan lap, mac dinh 1
        OUTPUT:
            - Mask co lai, nho lai.
    """
    kernel = choose_kernel(size_kernel, shape_kernel)
    return cv2.erode(mask, kernel, iterations = iterations)


def opening(mask, size_kernel=3, shape_kernel="elip"):
    """ 
        chu J co dom den (den:0, trang:255)
        Xoa nhung dom nho lom dom ngoai hinh (dom nho co gia tri bang gia tri vat the ).
        
        INPUT: 
            - mask: dang shape (w*h)
            - size_kernel: kich thuoc kernel. Default = 3
            - shape_kernel: 1 trong "elip", "rec", "cross", "custom". Default = elip
        OUTPUT:
            - Mask het lom dom.
    """
    kernel = choose_kernel(size_kernel, shape_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def closing(mask, size_kernel=3, shape_kernel="elip"):
    """
        Xoa nhung lo nho lom dom trong hinh (cac lo nho nay co gia tri khac vs gia tri vat the).
        
        INPUT: 
            - mask: dang shape (w*h)
            - size_kernel: kich thuoc kernel. Default = 3
            - shape_kernel: 1 trong "elip", "rec", "cross", "custom". Default = elip
        OUTPUT:
            - Mask het lo nho.
    """
    kernel = choose_kernel(size_kernel, shape_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def gradient(mask, size_kernel=3, shape_kernel="elip"):
    """
        Xoa phan trong cua vat the (Giong kieu seletom tim khung cua 1 cai duong).
        
        INPUT: 
            - mask: dang shape (w*h)
            - size_kernel: kich thuoc kernel. Default = 3
            - shape_kernel: 1 trong "elip", "rec", "cross", "custom". Default = elip
        OUTPUT:
            - Mask xoa phan trong cua vat the.  
    """
    kernel = choose_kernel(size_kernel, shape_kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)