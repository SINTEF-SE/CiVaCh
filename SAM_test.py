import numpy as np

import FastSam
import cv2

import sam
import pickle
import os.path
import math


# TUNABLE PARAMETERS
NOMINAL_IM_SIZE = 800
GAUSS_BLUR_SIGMA = 3
#LAPLACIAN_KSIZE = 5

SAM_MINIMUM_AREA = 2000
SAM_NO_OF_SEGMENTS = 20     # not used
SAM_POINTS_PER_SIDE = 20
SAM_POINTS_PER_BATCH = 100
SAM_PRED_IOU_THRESH = 0.92
SAM_STABILITY_SCORE_THRESH = 0.8
SAM_MIN_MASK_REGION_AREA = 500

FASTSAM_NO_OF_SEGMENTS = 20

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


#--------- Show detected segments ---------------
def show_anns(Title, anns):
    if len(anns) == 0:
        return
    
    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    img = np.ones((anns[0]['segmentation'].shape[0], anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in anns:
        m = ann['segmentation']
        size = ann['area']
        if (size > 2000):
            color_mask = np.concatenate([np.random.random(3), [0.6]])
            img[m] = color_mask
                    
    cv2.imshow(Title, img)
    cv2.waitKey(0)


#------------  MAIN -----------------
method = 'fSAM'      #SAM or FASTSAM
imDir = 'Datasett/Ostbo2'           #image directory

dirlist = os.listdir(imDir)
fileList = os.listdir(imDir)
noFiles = len(fileList)

#   Loop through all files in directory
for fName in dirlist:

    imName = imDir + "/" + fName
      
    truncNames = imName.rsplit('/')
    truncItems = len(truncNames)
    fileName = truncNames[truncItems-1]
    fExt = fileName.rsplit('.')
    fileName = fileName.replace(fExt[1], 'msk')
    #print(fileName)

    im = cv2.imread(imName)
    img = normalizeImage(im, NOMINAL_IM_SIZE, GAUSS_BLUR_SIGMA)      # Reduce image size to same area as 800 x 450 pixels (if neccessary)
    size = img.size /3
 
    #cv2.imshow('Bilde', img)
    #cv2.waitKey(100)
    bgrIm = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #----- Check segmentation method -------------
    if method == "SAM":
        # Full segmentation method
        #--- Check if masks are stored in file --------
        storedMasks = 0
        if (os.path.isfile('./masks/' + fileName)):
            with open ('./masks/' + fileName, 'rb') as fp:
                masks = pickle.load(fp)
            storedMasks = 1 
            #print('Using', len(masks), 'masks stored in file')
        
        if (storedMasks == 0):

            #----- SEGMENTATION ---------------
            #print('Starting segmentation .......')
            
            masks, totalMasks = sam.grid_segment(
                image=bgrIm,
                SAM_NO_OF_SEGMENTS=SAM_NO_OF_SEGMENTS,
                SAM_POINTS_PER_SIDE=SAM_POINTS_PER_SIDE,
                SAM_POINTS_PER_BATCH=SAM_POINTS_PER_BATCH,
                SAM_PRED_IOU_THRESH=SAM_PRED_IOU_THRESH,
                SAM_STABILITY_SCORE_THRESH=SAM_STABILITY_SCORE_THRESH,
                SAM_MIN_MASK_REGION_AREA=SAM_MIN_MASK_REGION_AREA, 
                SAM_MIN_AREA=SAM_MINIMUM_AREA
                )

            #print('Segmentation ready in', int(tStop_segment-tStart_segment + 0.5), 'seconds')
            #print('Found', totalMasks, 'objects, keeping', len(masks), 'objects above minimum size for post processing')

            #Write segmented masks to file for later use
            with open('./masks/' + fileName, 'wb') as fp:
                pickle.dump(masks, fp)
        
        show_anns('Original Masks', masks)

    else:
        # Fast segmentation method
        fsmasks, totalMasks = FastSam.grid_segment(bgrIm, FASTSAM_NO_OF_SEGMENTS)
        fsmasks = np.array(fsmasks) > 0 #convert from float to bool
        
        #create same structure as SAM masks
        masks = []
        for m in fsmasks:
            myMask = {}
            myMask['segmentation'] = m
            masks.append(myMask)
            
        FastSam.show_anns('FastSAM Original Masks', masks)

        #Remove small objects below minimum size
        minSize = 1     # Minimum size in percent of total picture size
        newMasks = FastSam.remove_smallObjects(masks, bgrIm, minSize) 
        masks = newMasks
        #FastSam.show_anns('Removed small Masks', masks)  
        