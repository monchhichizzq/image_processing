from PIL import ImageGrab
import numpy as np
import cv2

image = ImageGrab.grab()#获得当前屏幕
width = image.size[0]
height = image.size[1]
print("width:", width, "height:", height)
print("image mode:",image.mode)
k=np.zeros((width,height),np.uint8)
fourcc = cv2.VideoWriter_fourcc(*'XVID')#编码格式
video = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 1, (width, height))


#输出文件命名为test.mp4,帧率为16，可以自己设置
while True:
    img_rgb = ImageGrab.grab()
    img_bgr=cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)#转为opencv的BGR格式
    video.write(img_bgr)
    cv2.imshow('imm', img_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()

