import numpy as np
import cv2
import Civ_utils as ut

def pipe_shape(img, carve_out, PIPE_HW_RATIO_LB, PIPE_HW_RATIO_UB): #wh_ratio_lb, wh_ratio_ub):
    gray = cv2.cvtColor(carve_out, cv2.COLOR_BGR2GRAY)

    # binary img
    thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)[1]
   # cv2.imshow('thresh', thresh)
   # cv2.waitKey(0)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
   # cv2.drawContours(img,contours,-1,(0,0,255),3) 
   # cv2.imshow('all', img)
   # cv2.waitKey(0)

    if (len(contours) > 1):
       box, height, width = ut.combine_contours(contours)
    else:
       # contour = contours[0] #if len(contours) == 2 else contours[2]
        rotrect = cv2.minAreaRect(contours[0])
        (center), (width,height), angle = rotrect
        box = cv2.boxPoints(rotrect)

    #big_contour = max(contours, key=cv2.contourArea)

    # find w and h of minAreaRect
    boxpts = np.int0(box)
    
    #img = cv2.drawContours(img, [boxpts], 0, (0,255,255), 3) #draw outline in image using proper color
    #cv2.imshow('box', img)
    #cv2.waitKey(0)

    hw_ratio = 0
    if width > height:
        hw_ratio = width/height
    else:
        hw_ratio = height/width
  
    if hw_ratio >= PIPE_HW_RATIO_LB and hw_ratio <= PIPE_HW_RATIO_UB:
        return True
    return False

def rope_check(
        carve_out, 
        template,
        TEMPL_MATCH_THRESH,
        TEMPL_MATCH_TOTAL_MATCHES_THRESH,
        TEMPL_MATCH_NO_SCALES,
        TEMPL_MATCH_ANGLE_RANGE,
        TEMPL_MATCH_ANGLE_INTERVAL,
        ):

    rope_match = template_match(
            carve_out,
            template,
            TEMPL_MATCH_THRESH,
            TEMPL_MATCH_TOTAL_MATCHES_THRESH,
            TEMPL_MATCH_NO_SCALES,
            TEMPL_MATCH_ANGLE_RANGE,
            TEMPL_MATCH_ANGLE_INTERVAL,
    )
        
    return rope_match

def black_pipe_check(avg_color, std, mean, PIPE_MEAN_THRESH, PIPE_STD_THRESH, BLACK_COLOR_LB, BLACK_COLOR_UB): # , black_lb, black_ub):
    black_pipe_color = all(cv2.inRange(avg_color, BLACK_COLOR_LB, BLACK_COLOR_UB) > 0)
    #print(avg_color, black_pipe_color)
    print('Black-pipe-check:', black_pipe_color, 'mean', mean, 'mean-thresh', PIPE_MEAN_THRESH, 'std', std, 'std-thresh', PIPE_STD_THRESH)
    return black_pipe_color and mean < PIPE_MEAN_THRESH and std < PIPE_STD_THRESH
    
def white_pipe_check(avg_color, std, mean, PIPE_MEAN_THRESH, PIPE_STD_THRESH,  WHITE_COLOR_LB, WHITE_COLOR_UB): # ,white_lb, white_ub)
    white_pipe_color = all(cv2.inRange(avg_color, WHITE_COLOR_LB, WHITE_COLOR_UB) > 0)
    print('white-pipe-check:', white_pipe_color, 'mean', mean, 'mean-thresh', PIPE_MEAN_THRESH, 'std', std, 'std-thresh', PIPE_STD_THRESH)
    return white_pipe_color and mean < PIPE_MEAN_THRESH and std < PIPE_STD_THRESH

def template_match(
        image, 
        template,
        TEMPL_MATCH_THRESH,
        TEMPL_MATCH_TOTAL_MATCHES_THRESH,
        TEMPL_MATCH_NO_SCALES,
        TEMPL_MATCH_ANGLE_RANGE,
        TEMPL_MATCH_ANGLE_INTERVAL):
    
    # litt rare losninger her
    #cv2.imshow('bilde',image)
    

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_norm = np.zeros(template_gray.shape)
    template_norm = cv2.normalize(template_gray, template_norm,0,255,cv2.NORM_MINMAX)
    template_gauss = cv2.GaussianBlur(template_norm, (3,3), 0)

    #image = cv2.GaussianBlur(image, (5,5), 0)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    img_norm = np.zeros(img_gray.shape)
    img_norm = cv2.normalize(img_gray, img_norm, 0,255,cv2.NORM_MINMAX)
    img_gauss = cv2.GaussianBlur(img_norm, (3,3), 0)
    #cv2.imshow('gaussian',img_gauss)
    #cv2.waitKey(0)

    maxRes, maxAngle = testAngles(1.0, 0.0, 10, template_gauss, img_gauss)
    #print('MaxRes', maxRes, 'at angle', maxAngle, 'and Scale 1.0')

    maxRes, maxScale = testScales(maxAngle, 1.0, 0.05, template_gauss, img_gauss)
    #print('MaxRes', maxRes, 'at Scale', maxScale, 'and Angle', maxAngle)

    maxRes, maxAngle = testAngles(maxScale, maxAngle, 1, template_gauss, img_gauss)
    #print('MaxRes', maxRes, 'at angle', maxAngle, 'and Scale', maxScale)

    maxRes, maxScale = testScales(maxAngle, 1.0, 0.05, template_gauss, img_gauss)
    #print('MaxRes', maxRes, 'at Scale', maxScale, 'and Angle', maxAngle)

    template_resized = template_gauss.copy()
    template_resized =  cv2.resize(template_gauss, (0,0), fx=maxScale, fy=maxScale)
    template_rotated = ut.rotate_image(template_resized, maxAngle)
    res = cv2.matchTemplate(img_gauss, template_rotated, cv2.TM_CCOEFF_NORMED)
    matches = np.where(res >= TEMPL_MATCH_THRESH)
    #print('Matches', len(matches[0]))
    #cv2.imshow('res',res)

    #res = cv2.matchTemplate(img_gauss, template_rotated, cv2.TM_CCOEFF_NORMED)
    #cv2.imshow('res',res)
    #if (maxRes >= 0.80 and maxScale > 0.1) or maxRes >= 0.85:
      #  print('TRUE')

    #cv2.waitKey()

    #if (maxRes >= 0.80 and maxScale > 0.1) or maxRes >= 0.85:
    if maxRes >= 0.80 :
        return 1
    
    return 0

    '''total_matches = 0
    for i in range(0, TEMPL_MATCH_NO_SCALES):
        scale = 1.0 + float(i*0.10)
        #scale = 0.02 #+ float(i*0.03)
        template_resized = template_gauss.copy()
        template_resized =  cv2.resize(template_gauss, (0,0), fx=scale, fy=scale)
        for j in TEMPL_MATCH_ANGLE_RANGE:
            template_rotated = ut.rotate_image(template_resized, j*(TEMPL_MATCH_ANGLE_INTERVAL))
            cv2.imshow('template',template_rotated)
            res = cv2.matchTemplate(img_gauss, template_rotated, cv2.TM_CCOEFF_NORMED)
            cv2.imshow('res',res)
            matches = np.where(res >= TEMPL_MATCH_THRESH)
            total_matches += len(matches[0])
            print('Scale:', scale, 'j:', j, 'maxValue', np.max(res),'matches:', len(matches[0]), 'total:', total_matches)
            cv2.waitKey(0)
            del template_rotated
        del template_resized
    del template
    if total_matches > TEMPL_MATCH_TOTAL_MATCHES_THRESH:
        return 1
    return 0'''

def testAngles(scale, baseAngle, deltaAngle, template, image):
    #scale template
    template_resized = template.copy()
    template_resized =  cv2.resize(template, (0,0), fx=scale, fy=scale)
    maxRes = 0
    maxAngle = baseAngle
    for j in range(-9, 9, 1):
        template_rotated = ut.rotate_image(template_resized, baseAngle + j*deltaAngle)
        res = cv2.matchTemplate(image, template_rotated, cv2.TM_CCOEFF_NORMED)
        maxVal = np.max(res)
        #print(j, 'Angle', baseAngle + j*deltaAngle, 'Scale', scale, 'Res', maxVal)
        if maxVal > maxRes:
            maxRes = maxVal
            maxAngle = baseAngle + j*deltaAngle
        del template_rotated

    return maxRes, maxAngle

def testScales(Angle, baseScale, deltaScale, template, image):
    #rotate template
    template_resized = template.copy()
    maxRes = 0
    maxScale = baseScale
    for j in range(1, 20, 1):
        myScale = baseScale*j*deltaScale +0.05
        template_resized =  cv2.resize(template, (0,0), fx=myScale, fy=myScale)
        template_rotated = ut.rotate_image(template_resized, Angle)
        res = cv2.matchTemplate(image, template_rotated, cv2.TM_CCOEFF_NORMED)
        maxVal = np.max(res)
        #print(j, 'Scale', myScale, 'Angle', Angle, 'Res', maxVal)
        if maxVal > maxRes:
            maxRes = maxVal
            maxScale = myScale
        del template_resized
        

    return maxRes, maxScale



# ---------------- findContours  -----------------------------------------------------
# Find close countours of an object
# image - image in which to find the contours
# minSize - minimum area of valid contour 
# Return array of contours greater than minSize
def getContours(image, minSize):     
    #imgray = np.copy(image)
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #apply bluring
    gaussMask = cv2.GaussianBlur(imgray, (3,3), 0)
    ret, thresh = cv2.threshold(gaussMask, 0, 255, cv2.THRESH_BINARY)
    Cont, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cont is all contours found

    Contours = []   #will hold only those contours greater than minSize
    for cnt in Cont:
        # if the size of the contour is greater than a threshold
        if  cv2.contourArea(cnt) > minSize:
            Contours.append(cnt)

    return Contours