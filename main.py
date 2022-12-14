import cv2
import numpy as np
#from skimage import io

img = cv2.imread("musk.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

#Cartoonifying the image
color = cv2.bilateralFilter(img, 3, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)
cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)

cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("musk-cartoon.png", cartoon)