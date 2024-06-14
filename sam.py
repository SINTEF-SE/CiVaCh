import numpy as np
import torch
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def load_model():
    sam_checkpoint = "./SAM/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cpu"
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam_model.to(device=device)
    return sam_model
    

def grid_segment(
        image,
        SAM_NO_OF_SEGMENTS,
        SAM_POINTS_PER_SIDE,
        SAM_POINTS_PER_BATCH,
        SAM_PRED_IOU_THRESH,
        SAM_STABILITY_SCORE_THRESH,
        SAM_MIN_MASK_REGION_AREA,
        SAM_MIN_AREA
):
    
    model = load_model()
    mask_generator = SamAutomaticMaskGenerator(
        model=model,
        points_per_side=SAM_POINTS_PER_SIDE,
        points_per_batch=SAM_POINTS_PER_BATCH, 
        pred_iou_thresh=SAM_PRED_IOU_THRESH, 
        stability_score_thresh=SAM_STABILITY_SCORE_THRESH,
        min_mask_region_area=SAM_MIN_MASK_REGION_AREA
    )
    masks = mask_generator.generate(image)
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    total_masks = len(sorted_masks)
    #TODO: remove mask less that minimum area
    i = len(sorted_masks) -1
    for m in reversed(sorted_masks):
        if (m['area'] < SAM_MIN_AREA):
            sorted_masks.pop(i)
        i -= 1


    #return sorted_masks[:SAM_NO_OF_SEGMENTS]
    return sorted_masks, total_masks