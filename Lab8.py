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

# Sobel Horizontal Edges
plt.subplot(1, 2, 1), plt.imshow(sobelHorizontal, cmap='gray')
plt.title('Sobel Horizontal Edges'), plt.xticks([]), plt.yticks([])

# Sobel Vertical Edges
plt.subplot(1, 2, 2), plt.imshow(sobelVertical, cmap='gray')
plt.title('Sobel Vertical Edges'), plt.xticks([]), plt.yticks([])

plt.show()
