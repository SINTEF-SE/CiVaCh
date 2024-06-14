import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import FastSam
import cv2
import Civ_utils as ut
import sam
import time
import cv_detectors as cvd
import pickle
import os.path
import colorFilters as cf
#import colorsys
import cv_Report as rep
from PIL import Image
#from ultralytics import FastSAM
#from ultralytics.models.fastsam import FastSAMPrompt
#import xlsxwriter as xl
import cv_excel as ex

# TUNABLE PARAMETERS
NOMINAL_IM_SIZE = 800
GAUSS_BLUR_SIGMA = 3

LAPLACIAN_KSIZE = 5
ROPE_MEAN_THRESH = 150
ROPE_STD_THRESH = 75
PIPE_MEAN_THRESH = 180 #135
PIPE_STD_THRESH = 100 #95

SAM_MINIMUM_AREA = 2000
SAM_NO_OF_SEGMENTS = 20     # not used
SAM_POINTS_PER_SIDE = 20
SAM_POINTS_PER_BATCH = 100
SAM_PRED_IOU_THRESH = 0.92
SAM_STABILITY_SCORE_THRESH = 0.8
SAM_MIN_MASK_REGION_AREA = 500

FASTSAM_NO_OF_SEGMENTS = 20

TEMPL_MATCH_THRESH = 0.80 #0.82
TEMPL_MATCH_TOTAL_MATCHES_THRESH = 400
TEMPL_MATCH_NO_SCALES = 4
TEMPL_MATCH_ANGLE_RANGE = range(-1,16)
TEMPL_MATCH_ANGLE_INTERVAL = 10

PIPE_HW_RATIO_LB = 3
PIPE_HW_RATIO_UB = 20

#BLACK_COLOR_LB = (0,5,25)
#BLACK_COLOR_UB = (105,60,160)
BLACK_COLOR_LB = (0,0,0)
BLACK_COLOR_UB = (40,110,200)
WHITE_COLOR_LB = (20,0,190)
WHITE_COLOR_UB = (179,40,255) 

#template = cv2.imread("./rope_templ5_small.png")
template = cv2.imread("./rope_temp_tore.png")
#template = cv2.imread("Diverse plastbilder/ropeTemplate.png")


COLOR_THRESH = 30

BLK_RANGE = [(0,0,0),(20,71,200)]
#BLK_RANGE = [range(0,20), range(0,71), range(0,200)]
BLU_RANGE = [(10,100,76),(22,255,255)]
#BLU_RANGE = (range(10,22), range(100,255), range(76,255))
BGN_RANGE = [(22,53,0),(37,255,255)]
#BGN_RANGE = (range(23,37), range(53,255), range(0,255))
GRN_RANGE = [(43,60,60),(84,255,255)]
#GRN_RANGE = (range(43,84), range(60,255), range(60,255))
YLW_RANGE = [(85,148,111),(105,255,255)]
#YLW_RANGE = (range(85,105), range(148,255), range(111,255))
YBW_RANGE = [(101,96,55),(113,173,255)]
#YBW_range = (range(101,113), range(96,173), range(55,255))
GBW_RANGE = [(103,0,0),(116,114,146)]
#GBW_RANGE = (range(103,116), range(0,114), range(0,146))
RED_RANGE = [(113,117,108),(125,255,255)]
#RED_RANGE = (range(113,125), range(117,255), range(108,255))
PUP_RANGE = [(126,50,50),(150,255,255)]
#PUP_RANGE = (range(126,150), range(50,255), range(50,255))
WHT_RANGE = [(100,0,152),(179,51,255)]
#WHT_RANGE = (range(100,179), range(0,51), range(152,255))

#------------  MAIN -----------------
method = 'SAM'      #SAM or FASTSAM
imDir = 'Datasett/HP'       #image directory
lastDir = imDir.split("/")[-1] #only text after /
testDir = 'Testresultater/' + lastDir
excelFile = testDir + "/CVtestResults.xlsx"        #Test result excel file name

#create results directory
try:  
    os.mkdir(testDir)  
except OSError as error: 
    pass 
    #print(error)


#create excel file
book, es = ex.createExcel(excelFile, imDir, method)
dirlist = os.listdir(imDir)
fileList = os.listdir(imDir)
noFiles = len(fileList)
#   Loop through all files in directory
i = 0
for fName in dirlist:

    imName = imDir + "/" + fName
    print(i,'/',noFiles, fName)

    tStart_main = time.perf_counter() 

    #print('Preprocessing image...')
    #fName = "Ostbo/20230227_090253.jpg"
    #fName = "Diverse plastbilder/Flytekule.jpg"
    #fName = "Ostbo/20230227_085905.jpg"
    #fName = "20230227_085934.jpg"
    #fName = "IMG_6976.jpg"
    #fName = "Ostbo2/IMG_6976.jpg"
    #fName = "Diverse plastbilder/ropeTemplate.png"
    #fName = "plastror/ror28.jfif"
      
    truncNames = imName.rsplit('/')
    truncItems = len(truncNames)
    fileName = truncNames[truncItems-1]
    fExt = fileName.rsplit('.')
    fileName = fileName.replace(fExt[1], 'msk')
    #print(fileName)


    im = cv2.imread(imName)
    img = ut.normalizeImage(im, NOMINAL_IM_SIZE, GAUSS_BLUR_SIGMA)
    #img = im
    size = img.size /3
    #print(size)
 
    #cv2.imshow('Bilde', img)
    #cv2.waitKey(100)
    bgrIm = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #Lap = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    #bgrIm = cv2.filter2D(bgrIm, -1, Lap)
 
    #cv2.imshow('sharp', bgrIm)
    #cv2.waitKey(0)
    
    #gray = cv2.cvtColor(bgrIm, cv2.COLOR_BGR2GRAY)
    #laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=LAPLACIAN_KSIZE)
    #abs = cv2.convertScaleAbs(laplacian)

    #----- Check segmentation method -------------
    tStart_segment = time.perf_counter()

    if method == "SAM":
        # Full segmentation method
        #--- Check if masks are stored in file --------
        storedMasks = 0
        if (os.path.isfile('./masks/' + fileName)):
            with open ('./masks/' + fileName, 'rb') as fp:
                masks = pickle.load(fp)
            storedMasks = 1 
            #print('Using', len(masks), 'masks stored in file')
            
        #fsmasks, fstotalMasks = FastSam.grid_segment(img, FASTSAM_NO_OF_SEGMENTS)
        
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

            #Write to file
            with open('./masks/' + fileName, 'wb') as fp:
                pickle.dump(masks, fp)
        
        #ut.show_anns('Original Masks', masks)

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
            
        #FastSam.show_anns('Fast Original Masks', masks)
        #print(masks)
        newMasks = FastSam.remove_smallObjects(masks, bgrIm, 5) 
        masks = newMasks
        #print(newMasks)
        #segIm = FastSam.show_anns('Removed small Masks', newMasks)  
        
    
    # ---- Decide color for each mask ----------------

    tStart_class = time.perf_counter()

    #print(len(masks))
    masks[:] = [m for m in masks if np.any(m['segmentation'])] # remove elements that are all black
    #print(len(masks))

    for m in masks:
        mask = m['segmentation']
        imask = mask>0
        carve_out = np.zeros_like(bgrIm, np.uint8)
        carve_out[imask] = bgrIm[imask] 
        
        perc, color, pipeColor = cf.decideColor(carve_out)
        m['color'] = color
        m['perc'] = perc

        #cv2.imshow('orig masks', carve_out) 
        #cv2.waitKey()

    #print('original mask number: ', len(masks))
    # ----------------   Separate overlapping masks -------------------
    ut.sepMasks(masks)
    #print('Number of separated masks: ', len(masks))
    #FastSam.show_anns('Seaparated Masks', masks)
    #cv2.waitKey()

    #----- DECIDE COLOR of segmented masks
    #print('Deciding color for each mask....', end =" ")

    combinedMasks = []
    if (len(masks) > 0):
        for m in masks:
            mask = m['segmentation']
            imask = mask>0
            carve_out = np.zeros_like(bgrIm, np.uint8)
            carve_out[imask] = bgrIm[imask] 

            #cv2.imshow('carve_out', carve_out) 
            #cv2.waitKey()

            pos, combinedContours = ut.getCenter(carve_out)

            if (combinedContours != 0):
                #cv2.drawContours(carve_out, combinedContours, -1, (0,0,255), 2)
                
                m['center'] = pos
                m['contours'] = combinedContours

                perc, color, pipeColor = cf.decideColor(carve_out)

                m['color'] = color
                m['perc'] = perc
                m['pollution'] = ut.get_Pollution(perc)
                #print(perc)
                #cv2.waitKey()
                combinedMasks.append(m)

                # check if mask is same color and inside larger mask
                for cm in combinedMasks[:-1]:
                    #check for same color
                    if (m['color'] == cm['color']):
                        cmCont = cm['contours'][0]
                        #cmCont = cmContours[0]
                        if(cv2.pointPolygonTest(cmCont, pos, False) > 0):
                            #remove last mask from array
                            combinedMasks.pop()
                            removed = True
                            break

                        
    #print('OK')
    #print('Number of combined masks: ', len(combinedMasks))

    #showing all masks
    if (len(combinedMasks) != 0):
        #ut.show_anns('Segemented masks', masks)
        segIm = FastSam.show_anns('combined masks', combinedMasks)
        #print('Number of segmented objects:', len(masks))
        #print('Number of combined segmented objects:', len(combinedMasks))
        #cv2.waitKey(0)

    tStop_segment = time.perf_counter()

    #-------- CLASSIFICATION ----------------
    #print('Starting classification......')


    classified_masks = []       # empty list

    if (len(combinedMasks) > 0):
        #id = 1
        for m in combinedMasks:
            mask = m['segmentation']
            imask = mask>0
            carve_out = np.zeros_like(bgrIm, np.uint8)
            carve_out[imask] = bgrIm[imask] 

            #cv2.drawContours(carve_out, Contours, -1, (0,255,0), 3)
            
            m['class'] = 'Ukjent'
            m['description'] = ''
                
            classified = False
            
            #  testing ++++++++++++++++++++

            #ut.peaksAndValleys(carve_out)

            # +++++++++ END

            #*********  Test for rør *****
            pl = ut.checkAllParallellLines(carve_out)
            if pl == True: # and (m['color'] == cf.BLK or m['color'] == cf.WHT):
                m['class'] = 'Rør'
                classified_masks.append(m)
                classified = True
                
            #***** Test for vannrør ******
            bl = ut.checkColorParallellLines(carve_out)
            if bl == True and (m['color'] == cf.BLK or m['color'] == cf.WHT):
                m['description'] = 'Vannrør'


             #******  Test for circle ********
            contour = m['contours']
            isCircle, cont = ut.circle_match(contour, carve_out)
            if (isCircle and m['class'] == 'Ukjent'):
                #cv2.drawContours(img, cont, -1, (0, 255, 0), 2) 
                #cv2.drawContours(img, contour, -1, (255, 0, 0), 2) 
                #cv2.imshow('Circles', img) 
                #cv2.waitKey(0) 

                m['class'] = 'Flytekule'
                classified_masks.append(m)
                classified = True

            #***** Test for Tau **************
            if cvd.rope_check(carve_out, template, TEMPL_MATCH_THRESH, TEMPL_MATCH_TOTAL_MATCHES_THRESH, TEMPL_MATCH_NO_SCALES, TEMPL_MATCH_ANGLE_RANGE, TEMPL_MATCH_ANGLE_INTERVAL) and (m['class'] == 'Ukjent'):
                #rope_masks.append(m)
                m['class'] = 'Tau'
                classified_masks.append(m)
                classified = True
                                    
            
            #****   Test for garn ******
            p = m['perc']
            #x = slice(0,10)
            #pp = p[x]
            '''if ut.yarn_check(p) and (m['color'] != cf.BLK or m['color'] != cf.WHT) and m['class'] == 'Ukjent':
                m['class'] = 'Garn'
                classified_masks.append(m)
                classified = True
            

                      
            #******  Test for GardinTang ********
            percTang = p[10]
            if (percTang >= 60 and m['class'] == 'Ukjent'):
                #print('percTang', percTang)
                m['class'] = 'GardinTare'
                classified_masks.append(m)
                classified = True'''


            perc = p
            #print('BLK:', perc[0], 'PUP:', perc[1], 'BLU:', perc[2], 'BGN:', perc[3], 'GRN:', perc[4], 'YLW:', perc[5], 'RED:', perc[6],'BEI:', perc[7], 'LYW', perc[8], 'WHT:', perc[9], 'Tare:', perc[10], 'Dirt:', perc[11] )
            #cv2.imshow('carve_out', carve_out)
            #cv2.waitKey(0)
            #if classified == True:
            #   print('Classified as', m['class'])
            

        #cv2.waitKey(0)

        #print('number of classified objects', len(classified_masks))
        tStop_class = time.perf_counter()
        #print('Classification ready in', int(tStop_class-tStart_class + 0.5), 'seconds')
        #print("Rope segments: ", len(rope_masks))
        #print("Pipe segments: ", len(black_pipe_masks) + len(white_pipe_masks))
        if (len(classified_masks) != 0):
            imgC = np.ones((classified_masks[0]['segmentation'].shape[0], classified_masks[0]['segmentation'].shape[1], 4))
            imgC[:,:,3] = 0

            ut.show_anns_color('Objekter', classified_masks, imgC)

            #cv2.imshow('test', img)
            #cv2.waitKey(0)
            imgC = imgC[:,:,:3]     #remove alpha channel
            imgC = imgC*255
            imgC = imgC.astype(np.uint8)

            addImg = cv2.addWeighted(img,0.3 ,imgC,0.7,0)
        else:
            imgC = np.copy(img)
            imgC[:,:] = 0
            imgC = imgC[:,:,:3]     #remove alpha channel
            imgC = imgC*255
            imgC = imgC.astype(np.uint8)
            addImg = cv2.addWeighted(img,0.3 ,imgC,0.7,0)

        #Draw contours around identified objects
        for m in classified_masks:
            c = m['contours']
            cv2.drawContours(addImg, c, -1, (0,0,0), 2)
        

        #Add id to each identified object
        id = 1
        for cm in classified_masks:
            #coord = cm['point_coords']
            pos = cm['center']
            cX, cY = pos
            cX = cX - 15
            cY = cY + 15
            #print ('Koordinater', coord, pos) 
            #id = cm['id']
            cm['id'] = id
            #x = int(coord[0][0] + 0.5)
            #y = int(coord[0][1] + 0.5)
            #cv2.putText(imgC, str(id), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3) #(b,g,r)
            if cm['color'] == cf.BLK:
                cv2.putText(addImg, str(id), (cX,cY), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2) #(b,g,r)
            else:
                cv2.putText(addImg, str(id), (cX,cY), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) #(b,g,r)

            id += 1
        #addImg = cv2.addWeighted(img,0.3 ,imgC,0.7,0)
        #print(classified_masks)
        #addImg = imgC
        #addImg = img + imgC
        #cv2.imshow('Added', addImg)
        #cv2.waitKey(0)
        #ut.show_anns_color('pipe', white_pipe_masks, [0.8, 0.8, 0.8], imgC)
        #ut.show_anns_color('rope', rope_masks, [0.7, 0, 0], imgC)
    else:
        print('No classified objects available')

    #print(mask) 
    #print('Masks:', len(masks))
    #cv2.imshow('Bilde', img)
    #print('image',img)
    cv2.imwrite('org.png', img)     # save original image
    cv2.imwrite('box.png', addImg)        #save segmented image
    segIm = segIm*255
    cv2.imwrite('segIm.png', segIm)     # save segmented image


    #----------------------- PDF Report-------------------------------------------------------

    pdf = rep.PDF() 
    #filename = 'tuto2.pdf'
    fExt = fName.rsplit('.')
    pdfName = fName.replace(fExt[1], 'pdf')
    filename = testDir + "/" + pdfName        #Test result PDF name

    #y_pos = pdf.printImage2('org.png', 'box.png')  # original image to the left and image with outlines to the right
    y_pos = pdf.printImage3('org.png', 'segIm.png', 'box.png')  # original image to the left and image with outlines to the right
    y_pos += 35
    pdf.set_y(y_pos)

    pdf.printHeadLine('Nummer', 'Dekning %', 'Type', 'Beskrivelse', 'Plasttype', 'Forurensning') 

    for m in classified_masks:
        #if method == "SAM":
        areaString = f'{(m["area"] / size * 100):.1f}'
        #else:
        #  areaString = 'Null'
    
        Plastic = 'PE'
        pdf.printObjectLine(str(m['id']), areaString, m['class'], m['description'], Plastic, str(m['pollution']), m['color'])

    pdf.output(filename)

    # -------------------------------------------------

    tStop_main = time.perf_counter()

    segTime = int((tStop_segment-tStart_segment)*100 + 0.5)/100
    classTime = int((tStop_class-tStart_class)*100 + 0.5)/100
    totTime = int((tStop_main-tStart_main)*100 + 0.5)/100

    #------------------ WRITE TO EXCEL file ----------------------------------------------------------------------------

    es.write(3+i, 0, fName)
    es.write(3+i, 11, segTime)
    es.write(3+i, 12, classTime)
    es.write(3+i, 13, totTime)

    i += 1
# END for loop 
    
book.close()
import subprocess
#subprocess.Popen(['/' + filename],shell=True)
cdir = os.getcwd()
os.startfile(cdir + '/' + filename)
#subprocess.Popen(['tuto2.pdf'], shell = True)


print('Segmentation time:', segTime, 'seconds')
print('Classification time:', classTime, 'seconds')
print('Total time:', totTime, 'seconds')

 