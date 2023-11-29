import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread(r'C:\Lab8\ATU.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator for horizontal edges (x-direction)
sobelHorizontal = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)

# Apply Sobel operator for vertical edges (y-direction)
sobelVertical = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)

# Plotting
plt.figure(figsize=(15, 5))

# Top Row: Original Image and Grayscale Version
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 3, 2), plt.imshow(imgGray, cmap='gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])

# Bottom Row: Sobel Horizontal and Vertical Edges
plt.subplot(1, 3, 3), plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])

plt.show()
