import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

image = cv2.imread('stam.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



base_path = os.path.dirname(os.path.abspath(__file__))
sam_checkpoint = os.path.join(base_path, '..', 'segment_anything_hq/pretrained_model/sam_vit_h_4b8939.pth')
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(image)

boxes = [[598, 724, 744, 824], [152, 789, 277, 913], [726, 700, 865, 783], [1103, 715, 1317, 799], [1392, 754, 1501, 847], [1659, 658, 1730, 701], [1443, 649, 1493, 687], [538, 679, 605, 732], [1719, 656, 1791, 700], [487, 633, 530, 671], [2230, 845, 2273, 963], [1183, 619, 1251, 647], [628, 679, 699, 730], [2251, 910, 2302, 1060]]
# input_boxes = np.array([[598, 724, 744, 824], [152, 789, 277, 913], [726, 700, 865, 783], [1103, 715, 1317, 799], [1392, 754, 1501, 847], [1659, 658, 1730, 701], [1443, 649, 1493, 687], [538, 679, 605, 732], [1719, 656, 1791, 700], [487, 633, 530, 671], [2230, 845, 2273, 963], [1183, 619, 1251, 647], [628, 679, 699, 730], [2251, 910, 2302, 1060]])

input_boxes = torch.tensor(boxes, device=predictor.device)  
transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
transformed_boxes = transformed_boxes.type(torch.int32)
masks, _, _ = predictor.predict_torch(
    point_coords=None,
    point_labels=None,
    boxes=transformed_boxes,
    multimask_output=False,
)
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
# show_mask(masks[0], plt.gca())
# show_box(input_box, plt.gca())
# plt.axis('off')
# plt.show()


plt.figure(figsize=(10, 10))
plt.imshow(image)
for mask in masks:
    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
for box in input_boxes:
    show_box(box.cpu().numpy(), plt.gca())
plt.axis('off')
plt.show()