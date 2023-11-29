import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread(r'C:\Lab8\ATU.jpg')  # Use the correct file path and extension

# Check if the image is loaded successfully
if img is None:
    print("Error: Could not open or read the image.")
else:
    print("Image loaded successfully.")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Plot the original and grayscale images side by side
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(122), plt.imshow(gray_img, cmap='gray'), plt.title('Grayscale')

# Show the plot
plt.show()
