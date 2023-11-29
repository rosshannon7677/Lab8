import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the original image
imgOrig = cv2.imread(r'C:\Lab8\ATU.jpg')

# Convert the original image to grayscale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Blur the grayscale image
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)

# Apply Sobel detector to the grayscale image
sobelx = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)
imgSobel = np.sqrt(sobelx**2 + sobely**2)

# Apply Canny detector to the grayscale image
imgCanny = cv2.Canny(imgGray, 100, 200)

# Plotting
plt.subplot(2, 1, 1), plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB))
plt.title('Original'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 1, 2), plt.imshow(imgGray, cmap='gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

plt.show()
