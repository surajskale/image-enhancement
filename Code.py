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

threshold_for_noise = 0.5
limit_for_blurryness = 100

# new file name for saving image


def getNewFileName(file_name, prefix):
    split_file_name = file_name.split('/')
    # print(split_file_name)

    split_file_name[-1] = prefix + split_file_name[-1]
    new_file_name = "/".join(split_file_name)

    # print("new_file_name ", new_file_name)

    return new_file_name


def getNewFileNameWithExtension(file_name, prefix, file_extension):
    split_file_name = file_name.split('/')

    # print(split_file_name, file_extension)

    # print(split_file_name[-1].split('.'))

    file_name_with_old_extension = split_file_name[-1].split('.')

    file_name_with_old_extension[-1] = file_extension

    file_name_with_new_extension = ".".join(file_name_with_old_extension)

    split_file_name[-1] = file_name_with_new_extension

    split_file_name[-1] = prefix + split_file_name[-1]

    new_file_name = "/".join(split_file_name)

    return new_file_name


def isGreyScale(file_name):
    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    print("len(image.shape)", len(image.shape), "\n")
    if(len(image.shape) < 3):
        return True
    elif len(image.shape) == 3:
        return False

# get histogram for file_name


def getHistogram(file_name):
    print(file_name)
    img = cv2.imread(file_name)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)

    new_file_name = getNewFileNameWithExtension(file_name, "hist_", "png")

    plt.savefig(new_file_name)

    return new_file_name


def getNoise(image):

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


def getBlurryness(image):
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


# function for clahe greyscale

def claheGreyscaleMethod(file_name):
    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    # plt.hist(cl_img.flat, bins=100, range=(100, 255))
    ret, thresh = cv2.threshold(
        cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    new_file_name = getNewFileName(file_name, "clahe_greyscale_")

    cv2.imwrite(new_file_name, thresh)

    return new_file_name


def CLAHE(file_name):
    print("started CLAHE() ")

    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    print(len(image.shape))

    print("Color Image")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    lab_planes = list(lab_planes)

    lab_planes[0] = clahe.apply(lab_planes[0])
    img = image
    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    new_file_name = getNewFileName(file_name, "clahe_")

    cv2.imwrite(new_file_name, image)

    print("stopped CLAHE()\n")

    return new_file_name


def ourMethod(file_name):
    print("started ourMethod() ")

    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    print(len(image.shape))

    print("Color Image")

    if getNoise(image) > threshold_for_noise:
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    if getBlurryness(image) < limit_for_blurryness:
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image = cv2.filter2D(image, -1, sharpen_kernel)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    lab_planes = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    lab_planes = list(lab_planes)

    lab_planes[0] = clahe.apply(lab_planes[0])
    img = image
    lab = cv2.merge(lab_planes)

    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    new_file_name = getNewFileName(file_name, "enhanced_")

    cv2.imwrite(new_file_name, image)

    print("stopped ourMethod()\n")

    return new_file_name


# for ours

def enhanceGreyscaleMethod(file_name):
    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    # denoising
    image = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)

    se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    image = cv2.divide(image, bg, scale=255)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(image)
    # plt.hist(cl_img.flat, bins=100, range=(100, 255))
    ret, thresh = cv2.threshold(
        cl_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    new_file_name = getNewFileName(file_name, "our_greyscale_")

    cv2.imwrite(new_file_name, thresh)

    return new_file_name


def traditionalMethodForGreyScale(file_name):
    image = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
    equalized_image = cv2.equalizeHist(image)
    new_file_name = getNewFileName(file_name, "traditional_greyscale_")
    cv2.imwrite(new_file_name, equalized_image)
    return new_file_name


def traditionalMethod(file_name):
    print("started traditionalMethod()")

    img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    new_file_name = getNewFileName(file_name, "traditional_")

    cv2.imwrite(new_file_name, img_output)

    print("stopped traditionalMethod()")

    return new_file_name


# enhanceColorImage("g.jpg")
# NOISE
# BLURRYNESS
# LUMINOSITY
# INTENSITY
