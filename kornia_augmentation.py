import io
import requests
import cv2
import kornia as K
import numpy as np
import torch
from kornia.augmentation import AugmentationSequential
from kornia.geometry import bbox_to_mask
from matplotlib import pyplot as plt


def download_image(url: str, filename: str = "") -> str:
    filename = url.split("/")[-1] if len(filename) == 0 else filename
    # Download
    bytesio = io.BytesIO(requests.get(url).content)
    # Save file
    with open(filename, "wb") as outfile:
        outfile.write(bytesio.getbuffer())

    return filename


url = "https://raw.githubusercontent.com/kornia/data/main/panda.jpg"
download_image(url)

def plot_resulting_image(img, bbox, keypoints):
    img_array = K.tensor_to_image(img.mul(255).byte()).copy()
    img_draw = cv2.polylines(img_array, bbox.numpy(), isClosed=True, color=(255, 0, 0))
    for k in keypoints[0]:
        img_draw = cv2.circle(img_draw, tuple(k.numpy()[:2]), radius=6, color=(255, 0, 0), thickness=-1)
    return img_draw


# Show image #
img_tensor = K.io.load_image("panda.jpg", K.io.ImageLoadType.RGB32)[None, ...]  # BxCxHxW
h, w = img_tensor.shape[-2:]


# Plot image with bounding box and keypoints #
# aug_list = AugmentationSequential(
#     K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1, p=1.0),
#     K.augmentation.RandomAffine(360, [0.1, 0.1], [0.7, 1.2], [30.0, 50.0], p=1.0),
#     K.augmentation.RandomPerspective(0.5, p=1.0),
#     data_keys=["input", "bbox", "keypoints", "mask"],
#     same_on_batch=False,
# )
# aug_list = AugmentationSequential(
#     K.augmentation.RandomVerticalFlip(p=1.0),
#     data_keys=["input", "bbox", "keypoints", "mask"],
#     same_on_batch=False,
# )

aug_list = AugmentationSequential(
    K.augmentation.RandomRotation(degrees=90, p=0.5),
    data_keys=["input", "bbox", "keypoints"],
    same_on_batch=False,
)


# The bboxes shape is (1, N, 4, 2) (because we write a polyline)
bbox1 = [[355, 10], [660, 10], [660, 250], [355, 250]]
bbox2 = [[400, 10], [700, 10], [700, 250], [400, 250]] # each bbox is 4 points (and each point is x y) 
bboxes = torch.tensor([[bbox1]])
keypoints = torch.tensor([[[465, 115], [545, 116], [355, 10], [660, 10], [660, 250], [355, 250]]])

img_out = plot_resulting_image(img_tensor, bboxes[0], keypoints)

plt.axis("off")
plt.imshow(img_out)
plt.show()

# Forward the augmentation on the image #
augmented_image, augmented_bboxes, augmented_keypoints = aug_list(img_tensor, bboxes.float(), keypoints.float())
img_out = plot_resulting_image(
    augmented_image,
    augmented_bboxes.int(),
    augmented_keypoints.int(),
)

plt.axis("off")
plt.imshow(img_out)
plt.show()

# Invers augmentation #
img_inversed, bboxes_inversed, keypoints_inversed = aug_list.inverse(augmented_image, augmented_bboxes, augmented_keypoints)
img_out = plot_resulting_image(
    img_inversed,
    bboxes_inversed.int(),
    keypoints_inversed.int(),
)

plt.axis("off")
plt.imshow(img_out)
plt.show()