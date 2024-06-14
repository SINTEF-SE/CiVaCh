from ultralytics import FastSAM
import numpy as np
import cv2
import Civ_utils as ut

def grid_segment(image, FASTSAM_NO_OF_SEGMENTS):
    retina_masks = True
    verbose = False
    device = 'cpu' #or 'cuda' if available
    model = FastSAM('/weights/FastSAM-x.pt')
    
    #conf and iou are adjustable parameters
    results = model(image, device=device , retina_masks=retina_masks, verbose=verbose, iou=0.1, conf=0.3) # iou=0.92, conf=0.4)
    #print (results)
    
    #Plotting results
    #prompt_process = FastSAMPrompt(image, results, device=device)
    #ann = prompt_process.everything_prompt()
    #prompt_process.plot(annotations=ann,output_path='test.jpg')

    masks = np.array([])
    if results[0].masks is not None:
        masks = np.array(results[0].masks.data.cpu())
        # SORTING LARGEST FIRST
        areas = np.sum(masks.reshape(masks.shape[0], -1), axis=1)
        sorted_indices = np.argsort(-areas)
        masks = masks[sorted_indices]
    else: 
        masks = np.array([])
    return masks[:FASTSAM_NO_OF_SEGMENTS], len(masks)

def show_anns(Title, masks):
    if len(masks) == 0:
        return
       
    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for mask in masks:
        m = mask['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.6]])
        img[m] = color_mask
        
    #cv2.imshow(Title, img)
    #cv2.waitKey(0)
    return img

def remove_smallObjects(masks, image, sizePercent):
    newMasks = []
    imSize = image.size /3
    minSize = imSize * sizePercent/100
    #minSize = sizePercent
    #print('minSize', minSize)

    for m in masks:
        mask = m['segmentation']  
        carve_out = np.zeros_like(image)
        carve_out[mask] = (255,255,255)
                        
        gray = cv2.cvtColor(carve_out, cv2.COLOR_BGR2GRAY)          
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        #thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)
    
        # find contours in the thresholded image
        cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        areas = []
        for c in cnts:
            area = cv2.contourArea(c)
            areas.append(area)

        maxArea = max(areas)
        #print(maxArea)

        newCnts = []
        newIm = np.zeros_like(image)
        totArea = 0
        for c in cnts:
            area = cv2.contourArea(c)
            
            if area >= minSize: #and area >= maxArea*0.8:
                totArea += area
                newCnts.append(c)

        cv2.drawContours(newIm, newCnts, -1, (255,255,255), -1)
        bw = cv2.cvtColor(newIm, cv2.COLOR_BGR2GRAY)   
        bw[:] = [x / 255 for x in bw]
        bw = np.array(bw, dtype=bool)
        myMask = {}
        myMask['segmentation'] = bw
        myMask['area'] = totArea
        newMasks.append(myMask)
    
    return newMasks  


