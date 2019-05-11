import cv2
import os
import colorsys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt




def get_velocity(V, X, X_pred):
    for i in range(len(X)):
        V[i] = np.sqrt((X[i][0] - X_pred[i][0]) ** 2 + (X[i][1] - X_pred[i][1]) ** 2)
    return V


def max_velocity(filenames_2, p):
    # number of file
    t = 0
    V_d = []
    # filename_2 为txt 文件
    for filename_2 in filenames_2:
        # Path to file_txt
        p_file_txt = os.path.join(p, filename_2)
        g_txt = np.loadtxt(open(p_file_txt, "rb"))
        g_txt= tranlate_to_center(g_txt)
        # 自己设定max_v的阀值 为的d(0,1)*2 或 3
        if g_txt[0].any() or g_txt[1].any() != 0:
            max_V = distance(g_txt[0], g_txt[1])
            V_d.append(max_V)
    # maximum velocity for one gesture
    maxV = np.mean(V_d)
    return maxV


# 将中心点1固定于一点图的中心 如果图的比例是（240，120），中心点为（180，60）
def tranlate_to_center(X):
    dx = X[1][0] * 3 - 180
    dy = X[1][1] * 2 - 60
    X_1 = np.zeros((X.shape))
    for i in range(len(X)):
        X_1[i][0] = (X[i][0] * 3 - dx)
        X_1[i][1] = (X[i][1] * 2 - dy)
    return X_1


def remove_zero_rows(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]


def zero_rows_complete(X, X_pre):
    nonzero_row_indice, _ = X.nonzero()
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    # print('nonzero_row_indice', nonzero_row_indice)
    # print('unique_nonzero_indice', unique_nonzero_indice)
    # 如果点到中心点的距离大于一个阀值，那么这个值归0
    # 这个阀值为 0，1 亮点的距离的8倍
    thresh_dis = distance(X[0], X[1]) * 8
    # print('thresh',thresh_dis)
    for i in range(len(X)):
        if X[1][0] == 0:
            X[1] = X_pre[1]
            if i not in nonzero_row_indice:
                X[i] = X[1]
            if distance(X[i], X[1]) > thresh_dis:
                X[i] = X[1]

        else:
            if i not in nonzero_row_indice:
                X[i] = X[1]
            if distance(X[i], X[1]) > thresh_dis:
                X[i] = X[1]

    return X


def zero_rows_complete_0(X):
    nonzero_row_indice, _ = X.nonzero()
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    # print('nonzero_row_indice', nonzero_row_indice)
    # print('unique_nonzero_indice', unique_nonzero_indice)
    # 如果点到中心点的距离大于一个阀值，那么这个值归0
    # 这个阀值为 0，1 亮点的距离的8倍
    thresh_dis = distance(X[0], X[1]) * 8
    # print('thresh',thresh_dis)
    for i in range(len(X)):
        if i not in nonzero_row_indice:
            X[i] = X[1]
        if distance(X[i], X[1]) > thresh_dis:
            X[i] = X[1]


    return X




# move legs and head
def remove_hang(X):
    # x = np.delete(X,[8,9,10,11,12,13,14,19,20,21,22,23,24], axis = 0)
    # body_25
    # x = np.delete(X, [8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24], axis=0)
    # pose_18 remove legs [8, 9, 10, 11, 12, 13]
    legs = [8, 9, 10, 11, 12, 13]
    x = np.delete(X, legs, axis=0)
    return x


def distance(p1, p2):
    d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
    return d


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径


def get_files(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        filenames.sort()
        dirnames.sort()
        for dirname in dirnames:
            p_1 = os.path.join(dirpath, dirname)
            for (dpath, dnames, dfiles) in os.walk(p_1):
                dnames.sort()
                print('dnames', dnames)
                for dname in dnames:
                    p = os.path.join(dpath, dname)
                    frameClone = np.ones((240, 360, 3), np.uint8) * 255
                    frameClone_all = np.ones((240, 360, 3), np.uint8) * 255
                    for (dirpath_2, dirnames_2, filenames_2) in os.walk(p):
                        # 重新排序00。。。10.。。。100.。。1000.。。
                        filenames_2.sort()
                        filenames_2_rearrange = []
                        for filename_2 in filenames_2:
                            if len(os.path.splitext(filename_2)[0]) == 2:
                                filenames_2_rearrange.append(filename_2)
                        for filename_2 in filenames_2:
                            if len(os.path.splitext(filename_2)[0]) == 3:
                                filenames_2_rearrange.append(filename_2)
                        for filename_2 in filenames_2:
                            if len(os.path.splitext(filename_2)[0]) == 4:
                                filenames_2_rearrange.append(filename_2)
                        for filename_2 in filenames_2:
                            if len(os.path.splitext(filename_2)[0]) == 5:
                                filenames_2_rearrange.append(filename_2)
                        # print('filenames_2_rearrange', filenames_2_rearrange)
                        gesture = []
                        gesture_o = []
                        gesture_x = []
                        gesture_y = []
                        t = 0
                        plt.ion()  # interactive mode on
                        plt.gca()
                        plt.figure(1)
                        plt.figure(2)
                        plt.figure(3)
                        plt.figure(4)
                        for file in filenames_2_rearrange:
                            max_mean_V = max_velocity(filenames_2_rearrange, p)
                            # fix_point 1 (180,60)

                            p_file_txt = os.path.join(p, file)
                            g_raw = np.loadtxt(open(p_file_txt, "rb"))
                            V = np.zeros(len(g_raw))
                            g_raw = zero_rows_complete_0(g_raw)
                            # remove the useless coordinates
                            # g = remove_hang(g)

                            # complete the missiong coordiantes with the previous data or the central point
                            if gesture_o:
                                g_raw = zero_rows_complete(g_raw, gesture_o[t - 1])
                                V = get_velocity(V, g_raw, gesture_o[t - 1])
                            gesture_o.append(g_raw)
                            # # 平移至中心点
                            g_txt = tranlate_to_center(g_raw)
                            if gesture:
                                # g_txt = zero_rows_complete(g_txt, gesture[t - 1])
                                g_txt = zero_rows_complete(g_txt,gesture[t - 1])
                            gesture_x.append(g_txt[:, 0])
                            gesture_y.append(g_txt[:, 1])
                            gesture.append(g_txt)


                            # R, G, B是 [0, 255]. H 是[0, 360]. S, V 是 [0, 1].
                            t += 1
                            # # save image
                            # dir_path = '/Users/babalia/Desktop/final_project/data/action_data/4_figures_matplot'
                            # dir_4_path = '/Users/babalia/Desktop/final_project/data/action_data/CNN_3parts_matplot'
                            # image_4_save = dir_4_path + '/' + str(dirname)
                            # image_save = dir_path + '/' + str(dirname) + '/' + str(dname)
                            # mkdir(image_save)
                            # mkdir(image_4_save)
                            #
                            # path_save_all_A = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            #     dirname) + 'A.png'
                            # path_save_all_B = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            #     dirname) + 'B.png'
                            # path_save_all_C = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            #     dirname) + 'C.png'
                            # print(path_save_all_C)
                            # path_save_all_D = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            #     dirname) + 'D.png'


                            x = np.array(gesture_x)
                            y = np.array(gesture_y)

                            #  身体分为3部分 中间， 左， 右
                            # print('max(V)',max(V))
                            # body = [0,1,14,15,16,17]
                            # left = [2,3,4,8,9,10]
                            # right = [5,6,7,11,12,13]
                            # 身体分为3部分 上中下 body hands legs 主要手在动
                            # 分别为 K1， K2， K3
                            body = [0, 1, 14, 15, 16, 17]
                            hands = [2, 3, 4, 5, 6, 7]
                            legs = [8, 9, 10, 11, 12, 13]
                            # hands_left = [2, 3, 4]
                            # hands_right = [4, 5, 6]

                            # A spatial information
                            # K3
                            plt.figure(1)
                            plt.scatter(x, -y, s=5, c=(0,0,0))
                            plt.title("A")
                            plt.xlim(0, 360)
                            plt.ylim(-240,0)
                            fig_A = plt.gcf()




                            # plt.draw()
                            # plt.pause(0.1)


                            # B temporarel information
                            # K3, k2, K1
                            Hue = t / (len(filenames_2)) * 360 + 0
                            colors_HSV = [Hue, 1, 1]
                            # red, green, blue = hsv2rgb(Hue, Saturation, Brightness)
                            red, green, blue = colorsys.hsv_to_rgb(Hue / 360, 1, 1)
                            colors_plot = (red, green, blue)
                            plt.figure(2)
                            plt.scatter(x, -y, s=2, c=colors_plot)
                            plt.title("B")
                            plt.xlim(0, 360)
                            plt.ylim(-240, 0)
                            fig_B = plt.gcf()

                            # plt.draw()
                            # plt.pause(0.1)



                            # C Spectrum coding of body parts
                            # K3
                            for i in range(len(g_txt)):
                                if i in body:
                                    Hue = 0
                                # K1
                                if i in hands:
                                    Hue = t / (len(filenames_2)) * 180 + 0
                                # K1 left, right
                                # if i in hands_left:
                                #     Hue = t / (len(filenames_2)) * 180 + 0
                                # if i in hands_right:
                                #     Hue = - t / (len(filenames_2)) * 180 + 180
                                # K2
                                if i in legs:
                                    Hue = 180 + t / (len(filenames_2)) * 180
                                colors_HSV = [Hue, 1, 1]
                                red, green, blue = colorsys.hsv_to_rgb(Hue / 360, 1, 1)
                                colors_plot = (red,green,blue)
                                plt.figure(3)
                                plt.scatter(g_txt[i,0], -g_txt[i,1], s=2, c=colors_plot)
                                plt.title('C')
                                plt.xlim(0, 360)
                                plt.ylim(-240, 0)
                                fig_C = plt.gcf()

                                # plt.draw()
                                # plt.pause(0.1)


                                # D joint velocity weighted saturation and brightness max_mean
                                if max_mean_V != 0:
                                    if i in body:
                                        Hue = 0
                                        Saturation = 0
                                        # b = bmax - i/n (bmax-bmin)
                                        Brightness = 1 - t / (len(filenames_2)) * (1 - 0)
                                    # K1
                                    if i in hands:
                                        Hue = t / (len(filenames_2)) * 180 + 0
                                        # print(i, V[i] / max_mean_V)
                                        if V[i] / max_mean_V > 1:
                                            Saturation = 1
                                            Brightness = 1
                                        else:
                                            Saturation = V[i] / max_mean_V * 1
                                            Brightness = V[i] / max_mean_V * 1
                                    # K2
                                    if i in legs:
                                        # print(i, V[i] / max_mean_V)
                                        Hue = 180 + t / (len(filenames_2)) * 180
                                        if V[i] / max_mean_V > 1:
                                            Saturation = 1
                                            Brightness = 1
                                        else:
                                            Saturation = V[i] / max_mean_V * 1
                                            Brightness = V[i] / max_mean_V * 1

                                    colors_HSV = [Hue, Saturation, Brightness]

                                    red, green, blue = colorsys.hsv_to_rgb(Hue / 360, Saturation, Brightness)
                                    colors_plot = (red, green, blue)
                                    plt.figure(4)
                                    plt.scatter(g_txt[i, 0], -g_txt[i, 1], s=2, c=colors_plot)
                                    plt.title('D')
                                    plt.xlim(0, 360)
                                    plt.ylim(-240, 0)
                                    # get the figure

                                    # plt.draw()
                                    # plt.pause(0.1)

                        # save image
                        dir_path = 'C:\\action_data\\KTH\\4_figures_matplot'
                        dir_4_path = 'C:\\action_data\\KTH\\CNN_3parts_matplot'
                        image_4_save = dir_4_path + '\\' + str(dirname)
                        image_save = dir_path + '\\' + str(dirname) + '\\' + str(dname)
                        mkdir(image_save)
                        mkdir(image_4_save)

                        path_save_all_A = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            dirname) + 'A.png'
                        path_save_all_B = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            dirname) + 'B.png'
                        path_save_all_C = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            dirname) + 'C.png'
                        print(path_save_all_C)
                        path_save_all_D = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(
                            dirname) + 'D.png'

                        fig_A.savefig(path_save_all_A, dpi=600)
                        frame_A = cv2.imread(path_save_all_A)

                        fig_B.savefig(path_save_all_B, dpi=600)
                        frame_B = cv2.imread(path_save_all_B)

                        fig_C.savefig(path_save_all_C, dpi=600)
                        frame_C = cv2.imread(path_save_all_C)

                        fig_D = plt.gcf()
                        fig_D.savefig(path_save_all_D, dpi=1000)
                        frame_D = cv2.imread(path_save_all_D)


path = 'C:\\action_data\\KTH\\txt_cnn'
get_files(path)
# X_h_k, y_h_k = get_files(path_hammer_k, label=1)



