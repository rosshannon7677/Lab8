import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread(r'C:\Lab8\ATU.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Apply Sobel operator for horizontal edges (x-direction)
sobelX = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)

# Apply Sobel operator for vertical edges (y-direction)
sobelY = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)

# Calculate the sum of Sobel X and Sobel Y
sobelSum = cv2.addWeighted(sobelX, 0.5, sobelY, 0.5, 0)

# Apply Canny edge detection
cannyThreshold = 100
cannyParam2 = 200
canny = cv2.Canny(imgGray, cannyThreshold, cannyParam2)

# Plotting
plt.figure(figsize=(10, 10))

# Original Image
plt.subplot(3, 2, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

# Grayscale Image
plt.subplot(3, 2, 2), plt.imshow(imgGray, cmap='gray')
plt.title('Grayscale'), plt.xticks([]), plt.yticks([])

# Sobel X
plt.subplot(3, 2, 3), plt.imshow(sobelX, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

# Sobel Y
plt.subplot(3, 2, 4), plt.imshow(sobelY, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# Sobel Sum (Combination of Sobel X and Sobel Y)
plt.subplot(3, 2, 5), plt.imshow(sobelSum, cmap='gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])

# Canny Edge Detection
plt.subplot(3, 2, 6), plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
