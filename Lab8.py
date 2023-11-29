# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread(r'C:\Lab8\ATU.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur with different kernel sizes
imgBlur_3x3 = cv2.GaussianBlur(imgGray, (3, 3), 0)
imgBlur_5x5 = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgBlur_9x9 = cv2.GaussianBlur(imgGray, (9, 9), 0)
imgBlur_13x13 = cv2.GaussianBlur(imgGray, (13, 13), 0)

# Plotting
plt.figure(figsize=(15, 10))

# Original Image
plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Grayscale Image
plt.subplot(2, 3, 2), plt.imshow(imgGray, cmap='gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])

# Gaussian Blur (3x3)
plt.subplot(2, 3, 3), plt.imshow(imgBlur_3x3, cmap='gray')
plt.title('Gaussian Blur (3x3)'), plt.xticks([]), plt.yticks([])

# Gaussian Blur (5x5)
plt.subplot(2, 3, 4), plt.imshow(imgBlur_5x5, cmap='gray')
plt.title('Gaussian Blur (5x5)'), plt.xticks([]), plt.yticks([])

# Gaussian Blur (9x9)
plt.subplot(2, 3, 5), plt.imshow(imgBlur_9x9, cmap='gray')
plt.title('Gaussian Blur (9x9)'), plt.xticks([]), plt.yticks([])

# Gaussian Blur (13x13)
plt.subplot(2, 3, 6), plt.imshow(imgBlur_13x13, cmap='gray')
plt.title('Gaussian Blur (13x13)'), plt.xticks([]), plt.yticks([])

plt.show()
