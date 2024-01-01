import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Create two simple images
# image1 = np.array([[0, 0, 0],
#                    [0, 1, 0],
#                    [0, 0, 0]])

image1= np.zeros([25,25])
image1[3:7,3:7]=1
image1[15:19,4:8]=1
image1[15:19,20:24]=1

image2 = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])


#  Calculate spatial correlation using correlate2d
correlation = correlate2d(image1, image2, mode='full')


# Display the images and the spatial correlation result
plt.subplot(131), plt.imshow(image1, cmap='gray'), plt.title('Image 1')
plt.subplot(132), plt.imshow(image2, cmap='gray'), plt.title('Image 2')
plt.subplot(133), plt.imshow(correlation, cmap='viridis'), plt.title('Spatial Correlation')
plt.show()