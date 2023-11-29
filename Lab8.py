# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread(r'C:\Lab8\ATU.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with different kernel sizes
imgBlur_5x5 = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgBlur_9x9 = cv2.GaussianBlur(imgGray, (9, 9), 0)

# Plotting
plt.figure(figsize=(10, 8))

# Top Row: Original Image and Grayscale Version
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 2), plt.imshow(imgGray, cmap='gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])

# Bottom Row: Image Filtered with Different Kernels
plt.subplot(2, 2, 3), plt.imshow(imgBlur_5x5, cmap='gray')
plt.title('Gaussian Blur (5x5)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4), plt.imshow(imgBlur_9x9, cmap='gray')
plt.title('Gaussian Blur (9x9)'), plt.xticks([]), plt.yticks([])

plt.show()
