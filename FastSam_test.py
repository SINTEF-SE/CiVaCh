
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPrompt
import numpy as np
import cv2
import Civ_utils as ut
import time
import FastSam


NOMINAL_IM_SIZE = 800
GAUSS_BLUR_SIGMA = 7
FASTSAM_NO_OF_SEGMENTS = 20


##### MAIN #####
tStart = time.perf_counter()

#IMAGE_PATH = "Ostbo/20230227_085918.jpg"
IMAGE_PATH = "Ostbo2/IMG_6976.jpg"
#IMAGE_PATH = "Ostbo/20230227_090304.jpg"
image = cv2.imread(IMAGE_PATH)
img = ut.normalizeImage(image, NOMINAL_IM_SIZE, GAUSS_BLUR_SIGMA)
size = img.size /3
cv2.imshow('original', img)  

masks, antall = FastSam.grid_segment(img, 20)

tStop = time.perf_counter()

print ('Tid: ', int(tStop-tStart + 0.5), 'sekund')
print('Antall masker: ', len(masks))

#FastSam.show_anns('Original Masks', masks)

for m in masks:
    imask = m>0
    carve_out = np.zeros_like(img, np.uint8)
    carve_out[imask] = img[imask] 
    cv2.imshow('test', carve_out) 
    cv2.waitKey()

#cv2.imshow("mask example", masks[0])
#cv2.waitKey(0)
