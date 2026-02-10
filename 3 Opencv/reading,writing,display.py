import numpy as np
import cv2

input = cv2.imread(r'C:\AVSCODE\Opencv\tiger.jpg')

cv2.imshow('input', input)
cv2.waitKey(0)
cv2.destroyWindows()

# How do we save images we edit in OpenCV?
cv2.imwrite('output.jpg', input)
cv2.imwrite('output.png', input)