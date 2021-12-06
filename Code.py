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

s_thr = 0.5

# new file name for saving image


def getNewFileNameForEnhancedImage(file_name, prefix):
    split_file_name = file_name.split('/')
    print(split_file_name)

    split_file_name[-1] = prefix + split_file_name[-1]
    new_file_name = "/".join(split_file_name) 

    # print("new_file_name ", new_file_name)

    return new_file_name

# testing getNewFileNameForEnhancedImage()
# a = getNewFileNameForEnhancedImage("E:/MLCollegeProject/Anup/taj.jpeg", "asdfs_")
# print(a)

def isGreyScale(file_name):
    image = cv2.imread(file_name)
    print("len(image.shape)", len(image.shape), "\n")
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

    return s_perc


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
    return fm


def enhanceColorImage(file_name):
    print("started enhanceColorImage() ")

    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    print(len(image.shape))

    print("Color Image")

    if isNoisy(image) > s_thr:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    if isBlurry(image) < 100:
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

    new_file_name = getNewFileNameForEnhancedImage(file_name, "enhanced_")

    cv2.imwrite(new_file_name, image)

    print("stopped enhanceColorImage()\n")

    return new_file_name

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def enhanceGreyscaleImage(file_name):
    image = cv2.imread(file_name)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    # plt.hist(cl_img.flat, bins=100, range=(100, 255))
    ret, thresh = cv2.threshold(
        cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    new_file_name = getNewFileNameForEnhancedImage(file_name, "greyscale_")

    cv2.imwrite(new_file_name, cl_img)


def traditionalMethod(file_name):
    print("started traditionalMethod()")

    img = cv2.imread(file_name)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    new_file_name = getNewFileNameForEnhancedImage(file_name, "traditional_")

    cv2.imwrite(new_file_name, img_output)

    print("stopped traditionalMethod()")
    
    return new_file_name


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
