import os
import numpy as np
import sklearn
from sklearn import svm
import sklearn.metrics as sm

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import itertools
from sklearn.externals import joblib
from PIL import Image
import cv2
import time
import codecs
import math
import colorsys


def get_velocity(V, X, X_pred):
    for i in range(len(X)):
        V[i] = np.sqrt((X[i][0] - X_pred[i][0]) ** 2 + (X[i][1] - X_pred[i][1]) ** 2)
    return V


def max_velocity(filenames_2, p):
    # the mean of distance (0,1)*2
    V_d = []
    # filename_2 为txt 文件
    for filename_2 in filenames_2:
        # Path to file_txt
        p_file_txt = os.path.join(p, filename_2)
        g_txt = np.loadtxt(open(p_file_txt, "rb"))
        g_txt= tranlate_to_center(g_txt)
        # print('g_txt',g_txt)
        if g_txt[0].any() or g_txt[1].any() != 0:
            # 自己设定max_v的阀值 为的d(0,1)*2 或 3
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
        # delta_X = g_txt[1][0]-180 / delta_y = g_txt[1][1]-60
        X_1[i][0] = (X[i][0] * 3 - dx)
        X_1[i][1] = (X[i][1] * 2 - dy)
    return X_1


def remove_zero_rows(X):
    # X is a scipy sparse matrix. We want to remove all zero rows from it
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    return X[unique_nonzero_indice]


def zero_rows_complete(X):
    nonzero_row_indice, _ = X.nonzero()
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    # print('nonzero_row_indice', nonzero_row_indice)
    # print('unique_nonzero_indice', unique_nonzero_indice)
    # 如果点到中心点的距离大于一个阀值，那么这个值归0
    # 这个阀值为 0，1 亮点的距离的8倍
    thresh_dis = distance(X[0], X[1]) * 10
    # print('thresh',thresh_dis)
    for i in range(len(X)):
        if i not in nonzero_row_indice:
            X[i] = X[1]
            # X[i] = X_pre[i]
            # if X[i][0] == 0:
            #     X[i] = X[1]
        else:
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

    # else:
    #     print("---  There is this folder!  ---")


def get_files(path):
    for (dirpath, dirnames, filenames) in os.walk(path):
        filenames.sort()
        dirnames.sort()
        for dirname in dirnames:
            p_1 = os.path.join(dirpath, dirname)
            for (dpath, dnames, dfiles) in os.walk(p_1):
                dnames.sort()
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
                        t = 0
                        frame_A = np.ones((240, 360, 3), np.uint8) * 255
                        frame_B = np.ones((240, 360, 3), np.uint8) * 255
                        frame_C = np.ones((240, 360, 3), np.uint8) * 255
                        frame_D = np.ones((240, 360, 3), np.uint8) * 255
                        for file in filenames_2_rearrange:
                            max_mean_V = max_velocity(filenames_2_rearrange, p)
                            # print('max_V',max_V)
                            # fix_point 1 (180,60)
                            frameClone_A = np.ones((240, 360, 3)) * 255
                            frameClone_B = np.ones((240, 360, 3)) * 255
                            frameClone_C = np.ones((240, 360, 3)) * 255
                            frameClone_D = np.ones((240, 360, 3)) * 255

                            p_file_txt = os.path.join(p, file)
                            g_raw = np.loadtxt(open(p_file_txt, "rb"))
                            V = np.zeros(len(g_raw))

                            # remove the useless coordinates
                            # g = remove_hang(g)

                            # complete the missiong coordiantes with the previous data or the central point
                            if gesture_o:
                                g_raw = zero_rows_complete(g_raw)
                                V = get_velocity(V, g_raw, gesture_o[t - 1])
                            gesture_o.append(g_raw)

                            # # 平移至中心点
                            g_txt = tranlate_to_center(g_raw)
                            if gesture:
                                # g_txt = zero_rows_complete(g_txt, gesture[t - 1])
                                g_txt = zero_rows_complete(g_txt)
                            gesture.append(g_txt)

                            # R, G, B是 [0, 255]. H 是[0, 360]. S, V 是 [0, 1].
                            t += 1

                            for i in range(len(g_txt)):
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
                                colors_RGB = [0, 0, 0]
                                colors_BGR = colors_RGB[::-1]
                                cv2.circle(frameClone_A, (int(g_txt[i, 0]), int(g_txt[i, 1])), 4, colors_BGR, -1,
                                           cv2.LINE_AA)
                                cv2.circle(frame_A, (int(g_txt[i, 0]), int(g_txt[i, 1])), 1, colors_BGR, -1,
                                           cv2.LINE_AA)

                                # B temporarel information
                                # K3, k2, K1
                                Hue = t / (len(filenames_2)) * 360 + 0
                                colors_HSV = [Hue, 1, 1]
                                # red, green, blue = hsv2rgb(Hue, Saturation, Brightness)
                                red, green, blue = colorsys.hsv_to_rgb(Hue / 360, 1, 1)
                                colors_RGB = [red * 255, green * 255, blue * 255]
                                colors_BGR = colors_RGB[::-1]

                                # print('colors', colors_RGB)
                                cv2.circle(frameClone_B, (int(g_txt[i, 0]), int(g_txt[i, 1])), 4, colors_BGR, -1,
                                           cv2.LINE_AA)
                                cv2.circle(frame_B, (int(g_txt[i, 0]), int(g_txt[i, 1])), 1, colors_BGR, -1,
                                           cv2.LINE_AA)

                                # C Spectrum coding of body parts
                                # K3
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
                                #  colorsys.rgb_to_hsv(30/255, 50/255, 160/255)
                                # (0.6410256410256411, 0.8125, 0.6274509803921569)
                                red, green, blue = colorsys.hsv_to_rgb(Hue / 360, 1, 1)
                                colors_RGB = [red * 255, green * 255, blue * 255]
                                colors_BGR = colors_RGB[::-1]

                                # print('colors', colors_RGB)
                                cv2.circle(frameClone_C, (int(g_txt[i, 0]), int(g_txt[i, 1])), 4, colors_BGR, -1,
                                           cv2.LINE_AA)
                                cv2.circle(frame_C, (int(g_txt[i, 0]), int(g_txt[i, 1])), 1, colors_BGR, -1,
                                           cv2.LINE_AA)

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
                                    colors_RGB = [red * 255, green * 255, blue * 255]
                                    colors_BGR = colors_RGB[::-1]
                                    cv2.circle(frameClone_D, (int(g_txt[i, 0]), int(g_txt[i, 1])), 4, colors_BGR, -1,
                                               cv2.LINE_AA)
                                    cv2.circle(frame_D, (int(g_txt[i, 0]), int(g_txt[i, 1])), 1, colors_BGR, -1,
                                               cv2.LINE_AA)

                            ###hstack()在行上合并
                            ####vstack()在列上合并
                            frameClone_1 = np.hstack((frameClone_A, frameClone_B))
                            frameClone_2 = np.hstack((frameClone_C, frameClone_D))
                            frameClone = np.vstack((frameClone_1, frameClone_2))
                            # cv2.imshow("1", frameClone)
                            # cv2.waitKey(50)
                            # save image
                            dir_path = 'C:\\action_data\\KTH\\4_figures_Vmax_d'
                            dir_4_path = 'C:\\action_data\\KTH\\CNN_3parts_Vmax_d'
                            image_4_save = dir_4_path + '\\' + str(dirname)
                            image_save = dir_path + '\\' + str(dirname) + '\\' + str(dname)
                            mkdir(image_save)
                            mkdir(image_4_save)
                            path_save = dir_path + '/' + str(dirname) + '/' + str(dname) + '/' + str(file) + '.jpg'
                            cv2.imwrite(path_save, frameClone)
                            # cv2.imshow("2", frame_A)
                        path_save_all_A = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(dirname) + 'A.jpg'
                        print(path_save_all_A)
                        cv2.imwrite(path_save_all_A, frame_A)
                        # cv2.waitKey(50)
                        # cv2.imshow("3", frame_B)
                        path_save_all_B = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(dirname) + 'B.jpg'
                        cv2.imwrite(path_save_all_B, frame_B)
                        # cv2.waitKey(50)
                        # cv2.imshow("4", frame_C)
                        path_save_all_C = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(dirname) + 'C.jpg'
                        cv2.imwrite(path_save_all_C, frame_C)
                        # cv2.waitKey(50)
                        # cv2.imshow("5", frame_D)
                        path_save_all_D = dir_4_path + '/' + str(dirname) + '/' + str(dname) + str(dirname) + 'D.jpg'
                        cv2.imwrite(path_save_all_D, frame_D)
                        # cv2.waitKey(50)





path = 'C:\\action_data\\KTH\\txt_aug'
get_files(path)
