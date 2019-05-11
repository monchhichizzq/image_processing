import numpy as np
import cv2
image = cv2.imread("C:\\action_data\\KTH\\CNN_3parts_Vmax_2d\\running\\25runningD.jpg")

cv2.imshow("Original",image)
cv2.waitKey(0)
(h,w) = image.shape[:2]
center = (w / 2,h / 2)

#旋转45度，缩放0.75
M = cv2.getRotationMatrix2D(center,180,1)#旋转缩放矩阵：(旋转中心，旋转角度，缩放因子)
rotated = cv2.warpAffine(image,M,(w,h))

# Flipped Horizontally 水平翻转
h_flip = cv2.flip(image, 1)

# # Flipped Vertically 垂直翻转
# v_flip = cv2.flip(image, 0)
#
# # Flipped Horizontally & Vertically 水平垂直翻转
# hv_flip = cv2.flip(image, -1)

# 图像叠加
alpha = 0.5
beta = 0.5
img = cv2.addWeighted(image, alpha, h_flip, beta, gamma=0)
cv2.imshow("Rotated by 180 Degrees",img)
cv2.waitKey(0)
