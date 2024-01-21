import numpy as np
from samgeo import SamGeo, tms_to_geotiff, get_basemaps
import matplotlib.pyplot as plt
import cv2
import numpy as np

### Get input image: ###
input_image_path = "satellite.tif"
input_image = cv2.imread("satellite.tif")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

### Present input image: ###
plt.imshow(input_image)

### Initialize SAM-Geo Module: ###
samgeo_module = SamGeo(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth",
    sam_kwargs=None,
)

### Generate Segmentation Prediction: ###
output_image_path = "/home/reem/Desktop/playing_with_concepts/segment.tif"
flag_output_each_mask_with_unique_value = False
samgeo_module.generate(
    source=input_image, output=output_image_path, batch=True, foreground=False, erosion_kernel=(3, 3), mask_multiplier=255
)

### Show output image (mask): ###
output_image = cv2.imread(output_image_path)
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
number_of_masks = len(np.unique(output_image)[1:])
plt.imshow(output_image*255)
plt.title('number of masks:  ' + str(number_of_masks))
plt.show()

### Present output mask overlaid over input image: ###
# alpha_composite = np.concatenate([input_image.astype(float), output_image*255], -2)/255
alpha_composite = (0.5*input_image.astype(float) + 0.5*output_image*255)/255
plt.imshow(alpha_composite); plt.show()





