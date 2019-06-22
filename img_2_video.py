
import numpy as np
import cv2
import os

import cv2
import os, sys
import glob
import re



# load
img = cv2.imread('/Users/babalia/Desktop/2017103002310629.jpg')
# shape=(height, width, channel)
h,w,c = img.shape
# show

cv2.imshow('window_title', img)
cv2.waitKey()
cv2.destroyAllWindows()

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')

# videoWriter = cv2.VideoWriter('/Users/babalia/Desktop/AImove实习_patch/vgg16/vgg_16_visual.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
#                              (576,768))

path ='/Users/babalia/Desktop/AImove实习_patch/r'

for (dirpath_2, dirnames_2, filenames_2) in os.walk(path):
    # 重新排序00。。。10.。。。100.。。1000.。。
    filenames_2.sort()
    filenames_2_rearrange = []
    print('file',filenames_2)
    for filename_2 in filenames_2:
        # print(os.path.splitext(filename_2)[0][0:len(os.path.splitext(filename_2)[0])-4])
        if len(os.path.splitext(filename_2)[0]) == 1:
            filenames_2_rearrange.append(filename_2)
    for filename_2 in filenames_2:
        if len(os.path.splitext(filename_2)[0]) == 2:
            filenames_2_rearrange.append(filename_2)
    for filename_2 in filenames_2:
        if len(os.path.splitext(filename_2)[0]) == 3:
            filenames_2_rearrange.append(filename_2)
    for filename_2 in filenames_2:
        if len(os.path.splitext(filename_2)[0][0:len(os.path.splitext(filename_2)[0])-4]) == 4:
            filenames_2_rearrange.append(filename_2)
    print('file_rearrange',filenames_2_rearrange)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter('/Users/babalia/Desktop/AImove实习_patch/vgg16/vgg_16_visual.avi', fourcc, 10, (768, 576))
    for file in filenames_2_rearrange:
        path = '/Users/babalia/Desktop/AImove实习_patch/r/png2jpg'
        input_source = os.path.join(path,file)
        print(input_source)
        frame = cv2.imread(input_source)
        frame = cv2.resize(frame, (500,500))
        cv2.imshow('x',frame)
        cv2.waitKey()
        cv2.destroyAllWindows()
        print(frame)
        videoWriter.write(frame)

    videoWriter.release()




# #      大图
# for (direcpath, direcnames, files) in os.walk(im_dir):
#     print(direcpath, direcnames, files)
#     files.sort()
#     print('files', files)
#     for file in files:  # 依次读入列表中的内容
#         if file.endswith('jpg'):  # 后缀名'jpg'匹对
#             actual_path = os.path.join(im_dir, file)
#             print(actual_path)
#             frame = cv2.imread(actual_path)
#             print(frame.shape)
#             videoWriter.write(frame)
# videoWriter.release()
# print('finish')

