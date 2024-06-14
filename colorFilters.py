import cv2
import numpy as np
from enum import Enum

red = (0,0,255)
yellow = (0,255,255)
darkYellow = (0,200,255)
lightYellow = (190,240,240)
blue = (255,90,0)
darkBlue = (140,90,90)
blueGreen = (235,255,10)
green = (0,255,0)
black = (0,0,0)
white = (255,255,255)

#Clolors BGR format
BLK = (0,0,0)
BLU = (245,70,10)
BGN = (177,214,41)
BGL = (229,231,202)
GRN = (58,200,37)
YLW = (10,220,240)
LYW = (195,247,248)
BEI = (117,183,213)
RED = (7,25,240)
PUP = (240,7,140)
WHT = (250,250,250)
TAR = (27,32,2)

#indexes to perc
ixBLACK, ixPURPLE, ixBLUE, ixBLUEGREEN, ixGREEN, ixYELLOW, ixRED, ixBEIGE, ixLYELLOW, ixWHITE, ixTARE, ixDIRT = range(12)
pr = False

#  ----------------- Define color filters ----------------------------------------
def color_filter(file, lower_bound, upper_bound, mode):
    # filter colors within the boundaries
    #bgr_im = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(file, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    #cv2.imshow('file', file)
    #cv2.imshow('hsv', hsv)
    #cv2.imshow('filtermask', mask)

   
    if (mode == 0):
        mask = cv2.bitwise_not(mask)

    res = cv2.bitwise_and(file,file, mask = mask)
    #gaussMask = cv2.GaussianBlur(res, (3,3), 0)
    #find number of pixels
    totPix = np.sum(file > 0) /3
    maskPix = np.sum(mask == 255)
    pixPercent = int(maskPix/totPix *100)
    #print('totalPix', totPix, 'maskPix', maskPix, 'Percent', pixPercent)
    #cv2.waitKey(0)
    return pixPercent, res


# ---------------- Black filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but green color
# Mode = 0 - Remove green color
def Black_filter(file, mode):
    lower_bound_1 = np.array([0, 0, 1])	 
    upper_bound_1 = np.array([20, 190, 191])  #[179, 255, 30]
    lower_bound_2 = np.array([150, 0, 0])	 
    upper_bound_2 = np.array([179, 255, 255])  #[179, 255, 30]
    if pr:
        print('Black')
    PP1, mask1 = color_filter(file, lower_bound_1, upper_bound_1, mode)
    PP2, mask2 = color_filter(file, lower_bound_2, upper_bound_2, mode)
    return PP1+PP2, mask1 + mask2

   # return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- Black Rope filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but green color
# Mode = 0 - Remove green color
def BR_filter(file, mode):
    lower_bound = np.array([100, 0, 0])	 
    upper_bound = np.array([170, 60, 130])
    return color_filter(file, lower_bound, upper_bound, mode)

# ---------------- White filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but green color
# Mode = 0 - Remove green color
def White_filter(file, mode):
    lower_bound = np.array([0, 0, 178])	 # (0, 0, 150)
    upper_bound = np.array([179, 51, 255]) #(80, 50, 200)
    if pr:
        print('White')
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- Gray brown filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but green color
# Mode = 0 - Remove green color
def Beige_filter(file, mode):
    lower_bound = np.array([67, 30, 44])	 # (0, 0, 150)
    upper_bound = np.array([118, 99, 247]) #(80, 50, 200)
    if pr:
        print('Beige')
    return color_filter(file, lower_bound, upper_bound, mode)

# ---------------- Green filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but green color
# Mode = 0 - Remove green color
def Green_filter(file, mode):
    lower_bound = np.array([40, 50, 0])	 
    upper_bound = np.array([68, 255, 255])
    if pr:
        print('Green')
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- Blue-Green filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but blue-green color
# Mode = 0 - Remove blue-green color
def BGN_filter(file, mode):
    lower_bound = np.array([22, 53, 0])	 
    upper_bound = np.array([37, 255, 255])
    if pr:
        print('Blue green')
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- Blue-Green Light filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but blue-green color
# Mode = 0 - Remove blue-green color
def BGLight_filter(file, mode):
    lower_bound = np.array([22, 0, 126])	 
    upper_bound = np.array([77, 96, 255])
    if pr:
        print('Blue green light')
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- Red filter -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but red color
# Mode = 0 - Remove red color
def Red_filter(file, mode):
    lower_bound = np.array([113, 117, 108])	 # 150
    upper_bound = np.array([125 , 255, 255])

    #lower_bound_2 = np.array([0, 80, 130])	 #130
    #upper_bound_2 = np.array([9 , 255, 255])
    #lower_bound = np.array([107, 100, 100])	 
    #upper_bound = np.array([127, 255, 255])
    #bgr_file = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
    #res, boxes, myRect = color_filter(bgr_file, lower_bound, upper_bound, mode, boxArea)
    #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    if pr:
        print('Red')
    return color_filter(file, lower_bound, upper_bound, mode)
    


# ---------------- Yellow filter -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but yellow color
# Mode = 0 - Remove yellow color
def Yellow_filter(file, mode):
    lower_bound = np.array([72, 110, 25])	 
    upper_bound = np.array([105  , 255, 255])
    if pr:
        print('Yellow')
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- LightYellow filter -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but yellow color
# Mode = 0 - Remove yellow color
def LY_filter(file, mode):
    lower_bound = np.array([97, 0, 151])	 
    upper_bound = np.array([106  , 72, 255])
    if pr:
        print('Light yellow')
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- Blue filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but blue color
# Mode = 0 - Remove blue color
def Blue_filter(file, mode):
    lower_bound = np.array([5, 100, 76])	 
    upper_bound = np.array([18, 255, 255])
    if pr:
        print('Blue')
    return color_filter(file, lower_bound, upper_bound, mode)



# ---------------- Purple filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but purple color
# Mode = 0 - Remove purple color
def Purple_filter(file, mode):
    lower_bound = np.array([126, 50, 50])	 
    upper_bound = np.array([150, 255, 255])
    if pr:
        print('purple')
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- Dark blue filter,  -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but Dark blue color
# Mode = 0 - Remove dark blue color
def DB_filter(file, mode):
    lower_bound = np.array([100, 0, 0])	 
    upper_bound = np.array([179, 32, 134])
    return color_filter(file, lower_bound, upper_bound, mode)


# ---------------- White Pipe filter -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but red color
# Mode = 0 - Remove red color
def WP_filter(file, mode):
   # lower_bound_1 = np.array([90, 12, 0])	 
   # upper_bound_1 = np.array([130 , 103, 148])

    lower_bound_2 = np.array([90, 0, 195])	 
    upper_bound_2 = np.array([179 , 50, 255])
    #lower_bound = np.array([107, 100, 100])	 
    #upper_bound = np.array([127, 255, 255])
    #bgr_file = cv2.cvtColor(file, cv2.COLOR_RGB2BGR)
    #res, boxes, myRect = color_filter(bgr_file, lower_bound, upper_bound, mode, boxArea)
    #res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
   # res1 = color_filter(file, lower_bound_1, upper_bound_1, mode)
    return color_filter(file, lower_bound_2, upper_bound_2, mode)
   

# ---------------- Black Pipe filter -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but red color
# Mode = 0 - Remove red color
def Gray_filter(file, mode):
    lower_bound = np.array([105, 0, 0])	 
    upper_bound = np.array([119 , 180, 206])
    return  color_filter(file, lower_bound, upper_bound, mode)
    

# ---------------- Dirt filter -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but red color
# Mode = 0 - Remove red color
def Dirt_filter(file, mode):
    lower_bound = np.array([62, 10, 0])	 
    upper_bound = np.array([116 , 150, 236])
    if pr:
        print('Dirt')
    return color_filter(file, lower_bound, upper_bound, mode)
    


# ---------------- GardinTare filter -----------------------------------------------------
# file - Original input image
# Mode = 1 - Remove everything but red color
# Mode = 0 - Remove red color
def GardinTare_filter(file, mode):
    lower_bound = np.array([14, 12, 0])	 
    upper_bound = np.array([98 , 110, 200])
    if pr:
        print('Tare')
    return color_filter(file, lower_bound, upper_bound, mode)
    

# ---------------- Decide color -----------------------------------------------------
# carve_out - Carved out piece from an image
# COLOR_THRESH = minimum percentage of color within filter
# return = main color of pice and the percentage of this color in the piece
def decideColor(carve_out):
    perc = np.zeros(12, dtype=np.int8)
    colors = [BLK, PUP, BLU, BGN, GRN, YLW, RED, BEI, LYW, WHT, TAR]
    #colorsText = ['Black', 'Purple', 'Blue', 'BlueGreen', 'Green', 'Yellow', 'Red', 'Beige', 'Light yellow', 'White', 'Tare']
    perc[ixBLACK],_ = Black_filter(carve_out, 1)
    perc[ixPURPLE],_ = Purple_filter(carve_out, 1)
    perc[ixBLUE],_ = Blue_filter(carve_out, 1)
    perc[ixBLUEGREEN],_ = BGN_filter(carve_out, 1)
    perc[ixGREEN],_ = Green_filter(carve_out, 1)
    perc[ixYELLOW],_ = Yellow_filter(carve_out, 1)
    perc[ixRED],_ = Red_filter(carve_out, 1)
    perc[ixBEIGE],_ = Beige_filter(carve_out, 1)
    perc[ixLYELLOW],_ = LY_filter(carve_out, 1)
    perc[ixWHITE],_ = White_filter(carve_out, 1)
    perc[ixTARE],_ = GardinTare_filter(carve_out, 1)

    maxIndex = perc.argmax()
    color = colors[maxIndex]     #decide the color as the one with highest percentage
    #colorText = colorsText[maxIndex]
    BlackArray = perc[ixBLACK:ixBLUEGREEN]
    #print(BlackArray)
    WhiteArray = perc[ixLYELLOW:ixWHITE]
    #print(WhiteArray)
    BlackPipePerc = sum(BlackArray)
    #print ('black', BlackPipePerc)
    WhitePipePerc = sum(WhiteArray)
    #print('white', WhitePipePerc)

    if (BlackPipePerc > WhitePipePerc):
        pipeColor = 'Black'
    else:
        pipeColor = 'White'
    
    perc[ixDIRT],_ = Dirt_filter(carve_out, 1)

    #print('BLK:', perc[0], 'PUP:', perc[1], 'BLU:', perc[2], 'BGN:', perc[3], 'GRN:', perc[4], 'YLW:', perc[5], 'RED:', perc[6],'BEI:', perc[7], 'LYW', perc[8], 'WHT:', perc[9], 'Tare:', perc[10], 'Dirt:', perc[11] )

    return perc, color, pipeColor


# ---------------- Decide color -----------------------------------------------------
# carve_out - Carved out piece from an image
# COLOR_THRESH = minimum percentage of color within filter
# return = main color of pice and the percentage of this color in the piece
def decideColor_old(carve_out, COLOR_THRESH):
    perc = Black_filter(carve_out, 1)
    if perc > COLOR_THRESH:
        color = BLK 
        print('black')
    else:    
        perc = Blue_filter(carve_out, 1)
        if perc > COLOR_THRESH:
            color =BLU  
            print('blue', perc)
        else:
            perc = BGN_filter(carve_out, 1)
            if perc > COLOR_THRESH:
                color = BGN  
                print('bluegreen')
            else:
                perc = Green_filter(carve_out, 1)
                if perc > COLOR_THRESH:
                    color = GRN  
                    print('green')
                else:
                    perc = BGLight_filter(carve_out, 1)
                    if perc > COLOR_THRESH:
                        color = BGL  
                        print('Light BlueGreen')
                    else:
                        perc = Yellow_filter(carve_out, 1)
                        if perc > COLOR_THRESH:
                            color = YLW  
                            print('yellow')
                        else:
                            perc = LY_filter(carve_out, 1)
                            if perc > COLOR_THRESH:
                                color = LYW  
                                print('Light yellow')
                            else:
                                perc = Beige_filter(carve_out, 1)
                                if perc > COLOR_THRESH:
                                    color = BEI  
                                    print('gray-brown')
                                else:
                                    perc = Red_filter(carve_out, 1)
                                    if perc > COLOR_THRESH:
                                        color= RED
                                        print('red')
                                    else:
                                        perc = Purple_filter(carve_out, 1)
                                        if perc > COLOR_THRESH:
                                            color = PUP
                                            print('purple')
                                        else:
                                            perc = White_filter(carve_out, 1)
                                            if perc > COLOR_THRESH:
                                                color = WHT
                                                print('white')
                                            else:
                                                color = (255,255,255)
                                                print('Unknown color')

    return color, perc


def getColor(color):
    if color == 'R':
        return red
    if color == 'W':
        return white
    if color == 'G':
        return green
    if color == 'B':
        return blue
    if color == 'BG':
        return blueGreen
    if color == 'Y':
        return yellow
    if color == 'BK':
        return black


def getSubImage(src, rect):
    # Get center, size, and angle from rect
    center, size, theta = rect
    center, size = tuple(map(int, center)), tuple(map(int, size))
    x, y = center
    h, w = size
   
    
    if ((theta == 0) or (theta == 90)) :
        #crop image to rectangle without rotation
        crop_img = src[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
        return crop_img
    
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))
    out = cv2.getRectSubPix(dst, size, center)
    return out

def getFillGrade(image, rects):
    i = 0
    fillGrade = []
    for rec in rects:
        crop = getSubImage(image, rec)  
        if (crop.size > 0):
            FG = int((crop.size - np.sum(crop == 0))/crop.size * 100)
        else:
            FG = 0
        
        fillGrade.append(FG)
        i += 1
        #cv2.imshow("Crop", crop)
        #cv2.waitKey(0)
        #cv2.destroyWindow("Crop")
          
    return fillGrade
