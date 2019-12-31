# original ref: https://medium.com/pixel-wise/detect-those-corners-aba0f034078b

import cv2
import numpy as np

GREEN = [0, 255, 0]
corner_color = GREEN

'''
Function : cv2.cornerHarris(image, blocksize, ksize, k)
Parameters are as follows :
1. image : the source image in which we wish to find the corners (grayscale)
2. blocksize : size of the neighborhood in which we compare the gradient 
3. ksize : aperture parameter for the Sobel() Operator (used for finding Ix and Iy)
4. k : Harris detector free parameter (used in the calculation of R)
'''

def harris_corners(image):
    
    #Converting the image to grayscale
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #Conversion to float is a prerequisite for the algorithm
    gray_img = np.float32(gray_img)
    
    # 3 is the size of the neighborhood considered, aperture parameter = 3
    # k = 0.04 used to calculate the window score (R)
    corners_img = cv2.cornerHarris(gray_img, 3, 3, 0.04)
    
    #Marking the corners in Green
    image[corners_img > 0.0001*corners_img.max()] = corner_color

    return image


'''
Function: cv2.goodFeaturesToTrack(image,maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
image – Input 8-bit or floating-point 32-bit, single-channel image.
maxCorners – You can specify the maximum no. of corners to be detected. (Strongest ones are returned if detected more than max.)
qualityLevel – Minimum accepted quality of image corners.
minDistance – Minimum possible Euclidean distance between the returned corners.
corners – Output vector of detected corners.
mask – Optional region of interest. 
blockSize – Size of an average block for computing a derivative covariation matrix over each pixel neighborhood. 
useHarrisDetector – Set this to True if you want to use Harris Detector with this function.
k – Free parameter of the Harris detector.
'''

def shi_tomasi(image):

    #Converting to grayscale
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    #Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners
    corners_img = cv2.goodFeaturesToTrack(gray_img, 500, 0.02, 10)
    
    corners_img = np.int0(corners_img)

    for corners in corners_img:       
        x, y = corners.ravel()

        #Circling the corners in green
        circle_size = 3
        cv2.circle(image, (x,y), circle_size, corner_color, -1)

    return image
