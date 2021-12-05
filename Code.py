import cv2
from matplotlib import pyplot as plt
import numpy as np
# try:
from PIL import Image 
# except ImportError:
#     import Image
from imutils import paths
import argparse
from scipy import ndimage

def isGreyScale(image):
    if(len(image.shape) < 3):
        return True
    elif len(image.shape) == 3:
        return False

def isNoisy(image):

    # Convert image to HSV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image.shape[0:2])

    # Just for visualization and debug; remove in final
    # plt.plot(s)
    # plt.plot([p * 255, p * 255], [0, np.max(s)], 'r')
    # plt.text(p * 255 + 5, 0.9 * np.max(s), str(s_perc))
    # plt.show()
    # Just for visualization and debug; remove in final

    # Percentage threshold; above: valid image, below: noise
    # print("Greyscale image")
    print("s_perc \n\n", s_perc)

    s_thr = 0.5
    return s_perc > s_thr 


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def isBlurry(image):
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--images", required=True,
    # help="path to input directory of images")
    # ap.add_argument("-t", "--threshold", type=float, default=100.0,
    # help="focus measures that fall below this value will be considered 'blurry'")
    # args = vars(ap.parse_args())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    
    print("fm \n\n", fm)
    if fm < 100 :
        return True
    return False

def enhanceColorImage(file_name) :
    print(file_name)
    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    print(len(image.shape))
    
    print("Color Image")
    
    if isNoisy(image) == True:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    if isBlurry(image) == True:
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, -1, sharpen_kernel)
    
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes = list(lab_planes)

    lab_planes[0] = clahe.apply(lab_planes[0])
    img = image
    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # new file name for saving image
    split_file_name = file_name.split('/')
    split_file_name[-1] = "enhanced_" + split_file_name[-1] 
    new_file_name = "/".join(split_file_name)
    
    cv2.imwrite(new_file_name, image)
    
    return new_file_name

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def enhanceGreyscaleImage(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    # plt.hist(cl_img.flat, bins=100, range=(100, 255))
    ret, thresh = cv2.threshold(
        cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("image.jpg", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# if isGreyScale(image) == True:
#     print("greyscale")
#     image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
#     enhanceGreyscaleImage(image)
# else:
#     print("Color Image")
#     if isNoisy(image) == True:
#         image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
#     if isBlurry(image) == True:
#         sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
#         image = cv2.filter2D(image, -1, sharpen_kernel)
#     enhanceColorImage(image)


# enhanceColorImage("g.jpg")
