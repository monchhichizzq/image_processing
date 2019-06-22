import numpy as np
import cv2
import os
import numpy as np



def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    else:
        print("---  There is this folder!  ---")




input = '/Users/babalia/Desktop/final_project/database/ucf_sports_actions/ucf_action/video'
txt_directory = '/Users/babalia/Desktop/final_project/database/ucf_sports_actions/ucf_action/txt'
output = '/Users/babalia/Desktop/final_project/database/ucf_sports_actions/ucf_action/output'

for (direcpath, direcnames, dirfiles) in os.walk(input):
    for direcname in direcnames:
        p = os.path.join(input, direcname)
        print('p',p)
        for (dirpath, dirnames, files) in os.walk(p):
            files.sort()
            print(len(files))
            s = 0
            for file in files:
                input_source = os.path.join(p,file)
                print('input_source',input_source)
                dir_output = os.path.join(output,direcname)
                print('dir_output',dir_output)
                mkdir(dir_output)
                video_output = os.path.join(output,direcname,file)
                print('output_video',video_output)

                if s < 10:
                    csv_path = os.path.join(txt_directory, direcname,'0'+str(int((s+1)/18)))
                elif s >= 10:
                    csv_path = os.path.join(txt_directory, direcname, str(int((s+1)/18)))
                s += 1
                print('txt', csv_path)
                mkdir(csv_path)

                # input_source = "/Users/babalia/Desktop/final_project/database/ucf_sports_actions/ucf action/Golf-Swing-Front/007/RF1-13588_70046.avi"
                # csv_path = '/Users/babalia/Desktop/final_project/database/ucf_sports_actions/ucf action/Golf-Swing-Front/golf_swing_front/007da/'
                cap = cv2.VideoCapture(input_source)
                # cap = cv2.VideoCapture(0)
                hasFrame, frame = cap.read()

                # Define the codec and create VideoWriter object


                while (cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == True:
                        # frame = cv2.flip(frame, 0)

                        cv2.imshow('frame', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    else:
                        break

                # Release everything if job is finished
                cap.release()

                cv2.destroyAllWindows()
