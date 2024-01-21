import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

from segment_anything import sam_model_registry, SamPredictor

### Present logical (binary) mask with a certain color on it: ###
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

### Read image: ###
image = cv2.imread("satellite.tif")

### Initialize SAM & SAM-predictor: ###
sam = sam_model_registry['vit_h'](checkpoint='/home/reem/Desktop/universal_labeler/models/sam_vit_h_4b8939.pth').to(device='cuda')
SAM_predictor = SamPredictor(sam)
SAM_predictor.set_image(image)

### Take a BB over the entire image as prompt into SAM: ###
boxes = [[0, 0, 1790, 1116]]
input_boxes = torch.tensor(boxes, device=SAM_predictor.device)

### Perform a transform which resizes the image so that it's longest size is a certain size(???): ###
transformed_boxes = SAM_predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
transformed_boxes = transformed_boxes.type(torch.int32)

### Predict using the SAM-predictor over the entire image: ###
masks, _, _ = SAM_predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)

### Present image and masks: ###
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
plt.show()
