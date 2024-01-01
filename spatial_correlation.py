import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d

# Create two simple images
image1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

image2 = np.array([[9, 8, 7],
                   [6, 5, 4],
                   [3, 2, 1]])

#  Calculate spatial correlation using correlate2d
correlation = correlate2d(image1, image2, mode='full')


# Display the images and the spatial correlation result
plt.subplot(131), plt.imshow(image1, cmap='gray'), plt.title('Image 1')
plt.subplot(132), plt.imshow(image2, cmap='gray'), plt.title('Image 2')
plt.subplot(133), plt.imshow(correlation, cmap='viridis'), plt.title('Spatial Correlation')
plt.show()