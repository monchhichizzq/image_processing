
import cv2
import imageio
import os
images = []
path ='/Users/babalia/Desktop/AImove实习_patch/vgg16/output'

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

    for filename in filenames_2_rearrange:
        print(filename)
        path = '/Users/babalia/Desktop/AImove实习_patch/vgg16/output'
        input_source = os.path.join(path, filename)
        images.append(imageio.imread(filename))
    imageio.mimsave('/Users/babalia/Desktop/AImove实习_patch/vgg16/gif.gif', images,duration=1)