import cv2
import corner_detectors
import copy 

width_size = 1080
height_size = 720

IDX_LAPTOP_WEBCAM = 0 # if laptop, 0 is the default in-camera 
vid_capturer = cv2.VideoCapture(IDX_LAPTOP_WEBCAM)
vid_capturer.set(cv2.CAP_PROP_FRAME_WIDTH, width_size)
vid_capturer.set(cv2.CAP_PROP_FRAME_HEIGHT, height_size)

while True:
    _, frame = vid_capturer.read()
    
    corner_img_harris = corner_detectors.harris_corners(copy.deepcopy(frame))
    corner_img_shitomasi = corner_detectors.shi_tomasi(copy.deepcopy(frame))
 
    corner_img = cv2.hconcat([corner_img_harris, corner_img_shitomasi])

    cv2.imshow('harris (left) vs. shi_tomasi (right)', corner_img)

    key_ascii = cv2.waitKey(1)
    if key_ascii == 27: # 27=ESC
        break

vid_capturer.release()
cv2.destroyAllWindows()
