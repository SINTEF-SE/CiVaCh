#  Utility functions for CiVaCh
import math
import cv2
import numpy as np
import colorFilters as cf

p1 = '1 - Svært lite'
p2 = '2 - Lite'
p3 = '3 - Middels'
p4 = '4 - Mye'
p5 = '5 - Svært mye'

# ---------------- normalizeImage  -----------------------------------------------------
# Normalize (scale) an image to an area cor responding to 1200*675 pixels
# image - image to normalize
# Return normalized image
def normalizeImage(image, size, sigma):
   # norm = 1200*675     # normalized area of image
    height = int(image.shape[0])
    width = int(image.shape[1])
    norm = size*int(size/1.7777777777)     # normalized area of image (ie. 800x450)
    #print('Image original size (H x W):',height, 'x', width)
    imArea = height*width
    if imArea <= norm:
        return image
    
    scale = math.sqrt(norm / imArea)
    sWidth = int(width* scale)
    sHeight = int(height * scale)
    dim = (sWidth, sHeight)
  
    # resize image
    img_gauss = cv2.GaussianBlur(image, (sigma,sigma), 0)
    img = cv2.resize(img_gauss, dim, interpolation = cv2.INTER_AREA)
    #print('Image new size (H x W):',sHeight, 'x', sWidth)
    return img

def show_anns(Title, anns):
    if len(anns) == 0:
        return
    
    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in anns:
        m = ann['segmentation']
        size = ann['area']
        if (size > 2000):
            #color = [x/255 for x in color]
            #color_mask = np.concatenate([color, [0.6]])
            color_mask = np.concatenate([np.random.random(3), [0.6]])
            img[m] = color_mask
            #cv2.imshow('mask', img)
            #print('perc', ann['perc'])
            #cv2.waitKey(0)
        
    #ax.imshow(img)
    cv2.imshow(Title, img)
    #cv2.waitKey(0)


def show_anns_color(text, anns, img):
    if len(anns) == 0:
        return
    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    #ax = plt.gca()
    #ax.set_autoscale_on(False)

    #img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    #img[:,:,3] = 0
    for ann in anns:
        m = ann['segmentation']
        color = ann['color']
        color = [x/255 for x in color]
        color_mask = np.concatenate([color, [0.6]])
        img[m] = color_mask
        
    #ax.imshow(img)
    #cv2.imshow(text, img)
    #cv2.waitKey(0)
    

def rotate_image(image, angle):
    cpy = image.copy()
    rotation_center = (image.shape[1] // 2, image.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
    rotated_image = cv2.warpAffine(cpy, rotation_matrix, (image.shape[1], image.shape[0]))
    return rotated_image

def calculate_average_color(image, imask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_color = np.average(hsv[imask], axis=0)
    std_color = np.std(hsv[imask], axis=0)
    var_color = np.var(hsv[imask], axis=0)
   
    h,s,v = avg_color
    h = int(h)
    s = int(s)
    v = int(v)
    avg_color = (h,s,v)
    return avg_color

def getColorBins(image, imask):
    colorBins = []
    #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    black_perc = cf.Black_filter(image, 1)
    colorBins.append(black_perc)
    white_perc = cf.White_filter(image, 1)
    colorBins.append(white_perc)

    return colorBins

    

def combine_contours(contours):
    i = 0
    all_corners = []
    for cont in contours:
        rotrect = cv2.minAreaRect(cont)
        (center), (width,height), angle = rotrect
        box = cv2.boxPoints(rotrect)
        for point in box:                      # append all 4 points to all_corners
            all_corners.append(point)
        i += 1
   
    # Find top, bottom, left and right points
    top = bottom = left = right = all_corners[0]
    for point in all_corners:
        x, y = point
        if (y >= bottom[1]): bottom = point
        if (y <= top[1]): top = point
        if (x <= left[0]): left = point
        if (x >= right[0]): right = point

    #Now we have the bottom, left, top and right corners, create new box
    bigBox = [bottom, left, top, right]
    #calculate width and height of bigBox
    xl = bottom[0] - right[0]
    yl = bottom[1] - right[1]
    length = math.sqrt(xl*xl + yl*yl)
    xw = bottom[0] - left[0]
    yw = bottom[1] - left[1]
    width = math.sqrt(xw*xw + yw*yw)

    if (width > length):
        dum = width
        width = length
        length = dum

    return bigBox, length, width

def get_Pollution(perc):
    p11 = perc[11]
    pMax = max(perc)
    pol = min(p11,100-pMax)
    if pol < 10 : return p1
    if pol < 20 : return p2
    if pol < 30 : return p3
    if pol < 40 : return p4
    if pol >= 40 : return p5

def getCenter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    whiteIm = gray.copy()
    # get (i, j) positions of all RGB pixels that are black (i.e. [0, 0, 0])
    white_pixels = np.where(whiteIm[:, :] != 0)  

    if(len(white_pixels[0]) > 100): 

        # set those pixels to white
        whiteIm[white_pixels] = [255]

        blurred = cv2.GaussianBlur(whiteIm, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        # find contours in the thresholded image
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        sumCnt = []

        list_of_pts = [] 
        for ctr in cnts:
            list_of_pts += [pt[0] for pt in ctr]
            
        ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
        ctr = cv2.convexHull(ctr)

        if (len(cnts) != 1):
            sumCnt.append(ctr)
        else:
            sumCnt = cnts
        
        #print('len cnts',len(cnts), 'len ctr', len(sumCnt))
        #print('cnts', cnts, 'ctr',sumCnt)

        # compute the center of the contour
        M = cv2.moments(ctr)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        #cv2.drawContours(image, sumCnt, -1, (0, 255, 0), 2)
        #cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        #cv2.putText(image, "center", (cX - 20, cY - 20),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # show the image
        #cv2.imshow("Image", image)
        #cv2.waitKey(0)

        pos = (cX, cY)
    else:
        pos = 0
        sumCnt = 0

    return pos, sumCnt

def checkAllParallellLines(image):
    img = image.copy()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    kernel_size = 9
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
        

    # process edge detection using Canny
    low_threshold = 50
    high_threshold = 200
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    #cv2.imshow('edges', edges)

    #cv2.waitKey(0)

    #Use HoughlinesP to get the lines
    #rho = 1  # distance resolution in pixels of the Hough grid
    #theta = np.pi / 180   # angular resolution in radians of the Hough grid
    #threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    #min_line_length = 100  # minimum number of pixels making up a line
    #max_line_gap = 15  # maximum gap in pixels between connectable line segments  # var 10
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    #lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=80, maxLineGap=15)
    #lines = cv2.HoughLines(edges, rho=0.7, theta=1*np.pi / 180, threshold=70)
    theta = []
    if lines is not None:
        #print(len(lines))
        #print(lines[0])
      
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                if (x1 == x2):
                    theta.append(90)
                else:
                    th = math.atan((y2-y1)/abs(x2-x1))*180/np.pi
                    theta.append(th)
            
    #check if two or more lines are parallell
    theta.sort()
    #print(theta)
    
    parallell = False
    for i in range(len(theta)-1):
        #print(abs(theta[i] - theta[i+1]))
        if ((theta[i] != theta[i+1]) and abs(theta[i+1] - theta[i]) <= 4.0):
            parallell = True

    diffLines = 0
    #find lines with angle difference close to 90 deg
    for i in range(len(theta)-1):
        for j in range(i+1,len(theta)-1):
            if 45 <= abs(theta[i] - theta[j]) <=135:
                diffLines += 1

    if diffLines > 0:
        parallell = False           
            
    #print('dif', diffLines)
                
    # Draw the lines on the  image
    #lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    #if lines is not None:
     #   print('No lines:', len(lines))
    #cv2.imshow('lines', lines_edges)
    #print(parallell)
    #cv2.waitKey(0)
    return parallell #parallell


def checkColorParallellLines(img):
    _, img_filtered = cf.Blue_filter(img, 1)
    #cv2.imshow('filter', img_filtered)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(img_filtered,cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

    # process edge detection using Canny
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    #Use HoughlinesP to get the lines
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180   # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 70  # minimum number of pixels making up a line
    max_line_gap = 10  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
    theta = []
    if lines is not None:
        #print(len(lines))
        #print(lines[0])
      
        for line in lines:
          
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                if (x1 == x2):
                    theta.append(90)
                else:
                    th = math.atan((y2-y1)/abs(x2-x1))*180/np.pi
                    theta.append(th)
            
    #check if two or more lines are parallell
    theta.sort()
    #print(theta)
    parallell = False
    for i in range(len(theta)-1):
        #print(abs(theta[i] - theta[i+1]))
        if ((theta[i] != theta[i+1]) and abs(theta[i+1] - theta[i]) <= 4.0):
            parallell = True

    # Draw the lines on the  image
    #lines_edges = cv2.addWeighted(img_filtered, 0.8, line_image, 1, 0)

    '''cv2.imshow('lines', lines_edges)
    print(parallell)
    cv2.waitKey(0)'''
    return parallell

def peaksAndValleys(img):
    bgrIm = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    newIm = np.zeros_like(bgrIm, np.uint8)
    height = int(img.shape[0])
    width = int(img.shape[1])

    for h in range(1,height-1):
        for w in range(1, width -1):
            dot = bgrIm[h,w]
            nearbyDots = {bgrIm[h-1,w-1], bgrIm[h-1,w], bgrIm[h-1,w+1], bgrIm[h,w-1], bgrIm[h,w+1], bgrIm[h+1,w-1], bgrIm[h+1,w], bgrIm[h+1,w+1]}
            maxDot = max(nearbyDots)
            if dot >= maxDot - 5:
                newIm[h,w] = 255
    
    blurIm = cv2.GaussianBlur(newIm,(1, 1),0)
    
    #cv2.imshow('Valley', blurIm) 
    #cv2.waitKey(0) 
    
    ret, thresh = cv2.threshold(blurIm, 0, 255, cv2.THRESH_BINARY)
    Cont, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #cont is all contours found
    for c in Cont:
        cv2.drawContours(img, c, -1, (255,0,0), 2)

    cv2.imshow('newIm', img)
    cv2.waitKey(0)
    
    return

# ---------------- getContourFillGrade  -----------------------------------------------------
# Get the fill grade of all contours in a B/W mask, in percent
# Mask - B/W mask of the image
# Contours - Artray of contours to search in
# Return fillgrade for each contour in an array
def yarn_check(perc):
    #print(perc)
    maxP = max(perc)
    maxIndex = perc.argmax()
    #print(maxP, maxIndex)
    if (50 <= maxP <= 85 and cf.ixPURPLE <= maxIndex <=cf.ixRED):
        blackPerc = perc[cf.ixBLACK]
        whitePerc = perc[cf.ixWHITE]
        dirt = perc[cf.ixDIRT]
        if 5 <= blackPerc <= 40 or 5 <= whitePerc <= 40 and dirt <=5:
            #print('TRUE')
            return True
        
    
    #print('FALSE')
    return False

def circle_match(cont, image):
    
    img = image.copy()
    (x,y), radius = cv2.minEnclosingCircle(cont[0])
    center = (int(x), int(y))

    testIm = np.zeros_like(img, np.uint8) 
    cv2.circle(testIm,center,int(radius),(0,255,0),-1) 
    
    testbgr = cv2.cvtColor(testIm,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(testbgr, (3, 3), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            
    #print('Area', circleRel, 'Perimeter', perRel)
        
    #cv2.drawContours(img, cont, -1, (255, 0, 0), 2) 
    
    angles = getMinimumDistance(center, radius, cont, img)

    #cv2.drawContours(img, cnts, -1, (0, 0, 255), 2) 
 
    #cv2.imshow('Contours', img) 
    
    #if (circleRel >= 0.80 and abs(1-perRel) <= 0.1):
    ret = False
    if angles >= 150:
        #print('TRUE', angles, radius)
        ret = True
        
    #cv2.waitKey(0) 

    return ret, cnts

def getMinimumDistance(center, radius, cont, img):
    MAXDIST = radius/10
    aggrAngle = 0
    for th in range(0, 359, 1):
        thr = th*np.pi/180
        x = center[0] + radius*np.cos(thr)
        y = center[1] + radius*np.sin(thr)
        cv2.circle(img,(int(x),int(y)),int(5),(0,255,0),-1) 
        dist = abs(cv2.pointPolygonTest(cont[0],(x,y),True))
        #print('DISTANCE', dist)
        if dist <= MAXDIST:
            aggrAngle  += 1
    #print('aggrAngle', aggrAngle)
    #cv2.drawContours(img, cont, -1, (255, 0, 0), 2) 
    #cv2.imshow('points', img) 
    #cv2.waitKey(0)
    return aggrAngle

def separateMasks(cont1, cont2, img):
    cv2.drawContours(img, cont1, -1, (0,255,0), -1)   # the outer largest mask
    cv2.drawContours(img, cont2, -1, (0,0,255), -1)   # the inner mask
    #remove the red color
    img[np.all(img == (0, 0, 255), axis=-1)] = (0,0,0)

    # Find contour of the resulting green object
    testbgr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(testbgr, (3, 3), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img, cnts, -1, (255, 0, 0), 2) 

    #cv2.imshow('Tore', img)
    #cv2.waitKey(0)

    return cnts


# ---------------- sepMasks  -----------------------------------------------------
# Separate masks of different color, ie. maks of one color inside another mask of different color
# Mask - Boolean mask
# 
# Return nothing (masks are changed inside function)   
def sepMasks(masks):
    i = 0
    remArray = []
    for m in masks[:-1]:
        seg = m['segmentation']
        col = m['color']
        j = 0
        for nm in masks[i+1:]:
            index = i + j
            nseg = nm['segmentation']
            ncol = nm['color']
            if (col != ncol):
                # if not same color, split masks
                overlap = np.logical_and(seg,nseg)
                inv_overlap = np.logical_not(overlap)
                newSeg = np.logical_and(seg,inv_overlap)
                seg = newSeg
            '''else:
                #if masks  have the same color
                overlap = np.logical_and(seg,nseg)
                if (overlap == nseg).any():
                    #remove nseg
                    remArray.append(index)

            j += 1 '''  
        
        m['segmentation'] = seg
        i += 1
    
    #print(remArray)
   
    #for i in reversed(remArray):
       # masks.pop(i)


# ---------------- mergeMasks  -----------------------------------------------------
# Merge masks of same color that are overlapping (merge sub-mask into main mask)
# Mask - Boolean mask
# 
# Return nothing (masks are changed inside function)   
#def mergeMasks(masks):

