import utils
import cv_detectors as cvd
import sam
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def segment_ropes_and_pipes(
        image,
        template, 
        model,
        SAM_NO_OF_SEGMENTS,
        SAM_POINTS_PER_SIDE,
        SAM_POINTS_PER_BATCH,
        SAM_PRED_IOU_THRESH,
        SAM_STABILITY_SCORE_THRESH,
        SAM_MIN_MASK_REGION_AREA,
        DOWNSCALE_MINSIZE,
        GAUSS_BLUR_SIGMA,
        LAPLACIAN_KSIZE,
        ROPE_MEAN_THRESH,
        ROPE_STD_THRESH,
        PIPE_MEAN_THRESH,
        PIPE_STD_THRESH,
        TEMPL_MATCH_THRESH,
        TEMPL_MATCH_TOTAL_MATCHES_THRESH,
        TEMPL_MATCH_NO_SCALES,
        TEMPL_MATCH_ANGLE_RANGE,
        TEMPL_MATCH_ANGLE_INTERVAL,
        BLACK_COLOR_LB,
        BLACK_COLOR_UB,
        WHITE_COLOR_LB,
        WHITE_COLOR_UB,
        PIPE_WH_RATIO_LB,
        PIPE_WH_RATIO_UB):

    image_dscaled = utils.downscale(image, DOWNSCALE_MINSIZE)
    image_dsc_cpy = image_dscaled.copy()
    image_dscaled = cv2.cvtColor(image_dscaled, cv2.COLOR_RGB2BGR)
   
    blur = cv2.GaussianBlur(image_dsc_cpy, (GAUSS_BLUR_SIGMA,GAUSS_BLUR_SIGMA),0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=LAPLACIAN_KSIZE)
    abs = cv2.convertScaleAbs(laplacian)
  
    masks = sam.grid_segment(
        image=blur,
        model=model,
        SAM_NO_OF_SEGMENTS=SAM_NO_OF_SEGMENTS,
        SAM_POINTS_PER_SIDE=SAM_POINTS_PER_SIDE,
        SAM_POINTS_PER_BATCH=SAM_POINTS_PER_BATCH,
        SAM_PRED_IOU_THRESH=SAM_PRED_IOU_THRESH,
        SAM_STABILITY_SCORE_THRESH=SAM_STABILITY_SCORE_THRESH,
        SAM_MIN_MASK_REGION_AREA=SAM_MIN_MASK_REGION_AREA, 
        )
    
    plt.figure(figsize=(20,20)) 
    plt.imshow(abs)

    plt.figure(figsize=(20,20)) 
    plt.imshow(image_dscaled)
    utils.show_anns(masks)
    plt.title("Segments_detected")

    rope_masks = []
    black_pipe_masks = []
    white_pipe_masks = []
    for m in masks:
        mask = m['segmentation']
        imask = mask>0
        std = np.std(abs[imask])
        mean = np.mean(abs[imask])
        avg_color = utils.calculate_average_color(image_dscaled, imask)
        carve_out = np.zeros_like(image_dscaled, np.uint8)
        carve_out[imask] = image_dscaled[imask]
        cv2.imshow('mask', carve_out)
        cv2.waitKey(0)
        if cvd.rope_check(image_dscaled, imask, avg_color, std, mean, template, ROPE_MEAN_THRESH, ROPE_STD_THRESH, TEMPL_MATCH_THRESH, TEMPL_MATCH_TOTAL_MATCHES_THRESH, TEMPL_MATCH_NO_SCALES, TEMPL_MATCH_ANGLE_RANGE, TEMPL_MATCH_ANGLE_INTERVAL, BLACK_COLOR_LB, BLACK_COLOR_UB, WHITE_COLOR_LB, WHITE_COLOR_UB):
            rope_masks.append(m)
        elif cvd.black_pipe_check(avg_color, std , mean, PIPE_MEAN_THRESH, PIPE_STD_THRESH, BLACK_COLOR_LB, BLACK_COLOR_UB) and cvd.pipe_shape(carve_out, PIPE_WH_RATIO_LB, PIPE_WH_RATIO_UB):
            black_pipe_masks.append(m)
        elif cvd.white_pipe_check(avg_color, std, mean, PIPE_MEAN_THRESH, PIPE_STD_THRESH, WHITE_COLOR_LB, WHITE_COLOR_UB) and cvd.pipe_shape(carve_out, PIPE_WH_RATIO_LB, PIPE_WH_RATIO_UB):
            white_pipe_masks.append(m)
    print("Rope segments: ", len(rope_masks))
    print("Pipe segments: ", len(black_pipe_masks) + len(white_pipe_masks))

    plt.figure(figsize=(20,20)) 
    plt.imshow(image_dscaled)
    utils.show_anns_color(black_pipe_masks, [0, 0.7, 0])
    utils.show_anns_color(white_pipe_masks, [0, 0, 0.7])
    utils.show_anns_color(rope_masks, [0.7, 0, 0])
    plt.title("Pipes and ropes")
    plt.show() 


 ############  MAIN  ############ 

# TUNABLE PARAMETERS
DOWNSCALE_MINSIZE = 800
GAUSS_BLUR_SIGMA = 7

LAPLACIAN_KSIZE = 5
ROPE_MEAN_THRESH = 150
ROPE_STD_THRESH = 75
PIPE_MEAN_THRESH = 135
PIPE_STD_THRESH = 95

SAM_NO_OF_SEGMENTS = 20
SAM_POINTS_PER_SIDE = 20
SAM_POINTS_PER_BATCH = 100
SAM_PRED_IOU_THRESH = 0.92
SAM_STABILITY_SCORE_THRESH = 0.8
SAM_MIN_MASK_REGION_AREA = 500

TEMPL_MATCH_THRESH = 0.82
TEMPL_MATCH_TOTAL_MATCHES_THRESH = 400
TEMPL_MATCH_NO_SCALES = 4
TEMPL_MATCH_ANGLE_RANGE = range(-1,4)
TEMPL_MATCH_ANGLE_INTERVAL = 45

PIPE_WH_RATIO_LB = 0.05
PIPE_WH_RATIO_UB = 0.26

BLACK_COLOR_LB = (0,5,25)
BLACK_COLOR_UB = (105,60,160)
WHITE_COLOR_LB = (15,10,170)
WHITE_COLOR_UB = (50,30,255)

# Load SAM
sam_checkpoint = "./SAM/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cpu"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam_model.to(device=device)

# image = cv2.imread("../civach_bilder/blue_test_images/rope/20230227_085934.jpg")
# image = cv2.imread("../civach_bilder/blue_test_images/rope/20230227_085940.jpg")
# image = cv2.imread("../civach_bilder/red_test_images/not_rope/20230227_090042.jpg")
# image = cv2.imread("../civach_bilder/Ostbo/20230227_085943.jpg")
# image = cv2.imread("../civach_bilder/test_images/pipes/20230227_090119(0).jpg")
# image = cv2.imread("../civach_bilder/test_images/pipes/IMG_6934.JPG")
# image = cv2.imread("../civach_bilder/test_images/pipes/IMG_6940.JPG")

image = cv2.imread("./Ostbo/20230227_085933.jpg")
image = cv2.imread("./Ostbo/20230227_090253.jpg")
#image = cv2.imread("test_images/tau_eks1.jpg") # feil på tau

template = cv2.imread("./rope_templ5.png")

segment_ropes_and_pipes(
    image=image,
    template=template,
    model=sam_model,
    SAM_NO_OF_SEGMENTS=SAM_NO_OF_SEGMENTS,
    SAM_POINTS_PER_SIDE = SAM_POINTS_PER_SIDE,
    SAM_POINTS_PER_BATCH = SAM_POINTS_PER_BATCH,
    SAM_PRED_IOU_THRESH = SAM_PRED_IOU_THRESH,
    SAM_STABILITY_SCORE_THRESH = SAM_STABILITY_SCORE_THRESH,
    SAM_MIN_MASK_REGION_AREA = SAM_MIN_MASK_REGION_AREA,
    DOWNSCALE_MINSIZE=DOWNSCALE_MINSIZE,
    GAUSS_BLUR_SIGMA=GAUSS_BLUR_SIGMA,
    LAPLACIAN_KSIZE=LAPLACIAN_KSIZE,
    ROPE_MEAN_THRESH = ROPE_MEAN_THRESH,
    ROPE_STD_THRESH = ROPE_STD_THRESH,
    PIPE_MEAN_THRESH = PIPE_MEAN_THRESH,
    PIPE_STD_THRESH = PIPE_STD_THRESH,
    TEMPL_MATCH_THRESH = TEMPL_MATCH_THRESH,
    TEMPL_MATCH_TOTAL_MATCHES_THRESH = TEMPL_MATCH_TOTAL_MATCHES_THRESH,
    TEMPL_MATCH_NO_SCALES = TEMPL_MATCH_NO_SCALES,
    TEMPL_MATCH_ANGLE_RANGE = TEMPL_MATCH_ANGLE_RANGE,
    TEMPL_MATCH_ANGLE_INTERVAL = TEMPL_MATCH_ANGLE_INTERVAL,
    BLACK_COLOR_LB = BLACK_COLOR_LB,
    BLACK_COLOR_UB = BLACK_COLOR_UB,
    WHITE_COLOR_LB = WHITE_COLOR_LB,
    WHITE_COLOR_UB = WHITE_COLOR_UB,
    PIPE_WH_RATIO_LB = PIPE_WH_RATIO_LB,
    PIPE_WH_RATIO_UB = PIPE_WH_RATIO_UB
    )

#test entire folder
# for filename in os.listdir("../civach_bilder/test_images/pipes"):
#         print(filename)
#         if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"):
#             image = cv2.imread("../civach_bilder/test_images/pipes/"+filename)
#             segment_ropes_and_pipes(
#             image=image,
#             template=template,
#             model=sam_model,
#             SAM_NO_OF_SEGMENTS=SAM_NO_OF_SEGMENTS,
#             SAM_POINTS_PER_SIDE = SAM_POINTS_PER_SIDE,
#             SAM_POINTS_PER_BATCH = SAM_POINTS_PER_BATCH,
#             SAM_PRED_IOU_THRESH = SAM_PRED_IOU_THRESH,
#             SAM_STABILITY_SCORE_THRESH = SAM_STABILITY_SCORE_THRESH,
#             SAM_MIN_MASK_REGION_AREA = SAM_MIN_MASK_REGION_AREA,
#             DOWNSCALE_MINSIZE=DOWNSCALE_MINSIZE,
#             GAUSS_BLUR_SIGMA=GAUSS_BLUR_SIGMA,
#             LAPLACIAN_KSIZE=LAPLACIAN_KSIZE,
#             ROPE_MEAN_THRESH = ROPE_MEAN_THRESH,
#             ROPE_STD_THRESH = ROPE_STD_THRESH,
#             PIPE_MEAN_THRESH = PIPE_MEAN_THRESH,
#             PIPE_STD_THRESH = PIPE_STD_THRESH,
#             TEMPL_MATCH_THRESH = TEMPL_MATCH_THRESH,
#             TEMPL_MATCH_TOTAL_MATCHES_THRESH = TEMPL_MATCH_TOTAL_MATCHES_THRESH,
#             TEMPL_MATCH_NO_SCALES = TEMPL_MATCH_NO_SCALES,
#             TEMPL_MATCH_ANGLE_RANGE = TEMPL_MATCH_ANGLE_RANGE,
#             TEMPL_MATCH_ANGLE_INTERVAL = TEMPL_MATCH_ANGLE_INTERVAL,
#             BLACK_COLOR_LB = BLACK_COLOR_LB,
#             BLACK_COLOR_UB = BLACK_COLOR_UB,
#             WHITE_COLOR_LB = WHITE_COLOR_LB,
#             WHITE_COLOR_UB = WHITE_COLOR_UB,
#             PIPE_WH_RATIO_LB = PIPE_WH_RATIO_LB,
#             PIPE_WH_RATIO_UB = PIPE_WH_RATIO_UB
#             )
