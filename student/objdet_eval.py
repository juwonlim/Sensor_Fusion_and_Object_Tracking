# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
#matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well     
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    
     # find best detection for each valid label 
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid: # exclude all labels from statistics which are not considered valid
            
            # compute intersection over union (iou) and distance between centers

            ####### ID_S4_EX1 START #######     
            #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            #현재 라벨의 바운딩 박스의 네 모서리를 추출. 이를 위해 라벨의 좌표 정보를 사용하여 폴리곤을 만들 수 있음.
            
            ## step 2 : loop over all detected objects : 모든 검출된 객체에 대해 반복문을 사용하여 비교

            ## step 3 : extract the four corners of the current detection :검출된 객체의 바운딩 박스의 네 모서리 추출
                
            ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z :라벨과 검출된 객체 간의 중심 거리(x, y, z 축)를 계산
                
            ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box :라벨과 검출된 객체 바운딩 박스 간의 Intersection Over Union(IOU)를 계산
                
            ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count :IOU가 min_iou 값을 초과하는 경우, 해당 정보를 저장하고 True Positive 카운트를 증가시킴
                          
            #step 1~6 요약 :라벨(실제 객체)과 검출된 객체(모델이 예측한 것)를 비교하여 중심 거리 및 IOU를 계산합니다.
                            #IOU가 설정된 임계값을 초과하면 해당 객체를 정확하게 탐지한 것으로 간주하고, True Positive로 카운트합니다.
                            #이 과정을 통해 모델이 얼마나 정확하게 객체를 탐지했는지 평가할 수 있습니다.              

            #######
            #라벨과 탐지된 객체 간의 IOU와 중심 거리 계산
            ## step 1 : extract the four corners of the current label bounding-box (라벨 바운딩 박스의 네 모서리 추출)
                        #라벨은 ground truth라고도 부르며, 실제 이미지나 데이터에서 사람이 직접 표시한 정답입니다.
                        #라벨에는 각 객체의 바운딩 박스 좌표가 포함되어 있습니다. 이 좌표는 사각형 형태로 객체의 위치를 나타내며, 네 개의 모서리로 정의됩니다.
                        #이 단계에서 라벨의 좌표를 사용하여 그 바운딩 박스의 네 모서리 좌표를 추출합니다.
                        #이 모서리들은 2D 공간에서 x, y 좌표 또는 3D 공간에서는 x, y, z 좌표를 가질 수 있습니다.
            label_corners = tools.compute_box_corners(label.box.center_x, label.box.center_y, label.box.width, label.box.length, label.box.heading) 
            #사람이 라벨링하고 box친 물체의 4점 좌표

            ## step 2 : loop over all detected objects (모든 검출된 객체에 대한 반복문 사용)
                       #검출된 객체는 모델이 예측한 바운딩 박스입니다.
                       #이 단계에서는 모든 검출된 객체에 대해 반복문을 돌면서, 라벨과 비교하게 됩니다.
            for det in detections:
                
                ## step 3 : extract the four corners of the current detection (검출된 객체 바운딩 박스의 네 모서리 추출)
                          #검출된 객체도 마찬가지로 바운딩 박스를 가지고 있습니다.
                          #이 단계에서, 모델이 예측한 검출된 객체 바운딩 박스의 네 모서리를 추출합니다.
                det_corners = tools.compute_box_corners(det[1], det[2], det[5], det[6], det[7]) #컴퓨터가 박스치며 추출한 것,4점 좌표
                
                ## step 4 : compute the center distance between label and detection bounding-box in x, y, and z (라벨과 검출된 객체 간의 중심 거리 계산)
                            #라벨 바운딩 박스와 검출된 바운딩 박스의 중심 좌표를 계산합니다.
                            #이때 x, y, z 축을 기준으로 두 바운딩 박스 중심 간의 거리를 계산합니다.
                            #예를 들어, x축에서는 두 중심의 x좌표 차이, y축에서는 y좌표 차이, z축에서는 z좌표 차이를 계산합니다.
                            #중심 거리를 계산하는 이유는 두 객체가 얼마나 가까이 있는지를 평가하기 위함입니다.

                dist_x = abs(label.box.center_x - det[1]) #인간이 만든 좌표 - 컴퓨터가 만든 좌표
                dist_y = abs(label.box.center_y - det[2])
                dist_z = abs(label.box.center_z - det[3])
                
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box (Intersection Over Union(IOU) 계산)
                           #IOU는 두 개의 바운딩 박스가 얼마나 겹치는지를 측정하는 지표입니다. **Intersection(교집합)**과 **Union(합집합)**의 비율로 정의됩니다.
                            #교집합은 두 바운딩 박스가 겹치는 영역입니다. 합집합은 두 바운딩 박스의 전체 영역을 의미합니다. 
                            #IOU가 높을수록 모델이 정확하게 객체를 탐지했다고 볼 수 있습니다.

                poly_label = Polygon(label_corners)
                poly_det = Polygon(det_corners)
                intersection_area = poly_label.intersection(poly_det).area
                union_area = poly_label.union(poly_det).area
                iou = intersection_area / union_area
                
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                   #Step 6: IOU가 임계값을 초과하면 정보 저장 및 True Positive(TP) 카운트 증가
                            #**True Positive(TP)**는 모델이 객체를 정확하게 탐지한 경우를 의미합니다.
                            #min_iou라는 임계값(threshold)이 설정되어 있으며, IOU가 이 임계값보다 크면 "정확하게 탐지했다"고 간주합니다.
                            #IOU가 임계값을 초과하는 경우, 그 정보를 저장하고 TP 카운트를 증가시킵니다.
                            #이때 저장되는 정보는 IOU, x축 거리, y축 거리, z축 거리입니다.
                if iou > min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])
                    true_positives += 1

            
            ####### ID_S4_EX1 END #######     
            
        # find best match and compute metrics
        if matches_lab_det:
            best_match = max(matches_lab_det,key=itemgetter(1)) # retrieve entry with max iou in case of multiple candidates   
            ious.append(best_match[0])
            center_devs.append(best_match[1:])


    ####### ID_S4_EX2 START #######     
    #######
    print("student task ID_S4_EX2")
    
    # compute positives and negatives for precision/recall
    
    #긍정 및 부정 탐지 평가
    ## step 1 : compute the total number of posi
    # 
    # 
    # 
    # 
    # tives present in the scene :현재 장면(scene)에서 총 Positive 객체의 수를 계산. 이는 유효한 라벨 수와 같다고 할 수 있음
    all_positives = sum(labels_valid)

    ## step 2 : compute the number of false negatives :False Negative(탐지되지 않은 객체)의 수를 계산. 이는 라벨 중 탐지되지 않은 객체 수임.
    false_negatives = all_positives - true_positives

    ## step 3 : compute the number of false positives:False Positive(잘못 탐지된 객체)의 수를 계산. 이는 라벨과 일치하지 않는 탐지 결과임
    false_positives = len(detections) - true_positives
    
    #######
    ####### ID_S4_EX2 END #######     
    
    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
    
    ####### ID_S4_EX3 START #######     
    #######    
    print('student task ID_S4_EX3')

    # Precision과 Recall 계산
    ## step 1 : extract the total number of positives, true positives, false negatives and false positives
    #:총 Positive 수, True Positive 수, False Negative 수, False Positive 수를 추출
    total_positives = sum([pn[0] for pn in pos_negs])
    total_true_positives = sum([pn[1] for pn in pos_negs])
    total_false_negatives = sum([pn[2] for pn in pos_negs])
    total_false_positives = sum([pn[3] for pn in pos_negs])
    
    ## step 2 : compute precision
    #Precision(정밀도)을 계산: Precision = True Positive / (True Positive + False Positive).
    precision = total_true_positives / (total_true_positives + total_false_positives) if (total_true_positives + total_false_positives) > 0 else 0.0


    ## step 3 : compute recall 
    # Recall(재현율)을 계산: Recall = True Positive / (True Positive + False Negative)
    recall = total_true_positives / (total_true_positives + total_false_negatives) if (total_true_positives + total_false_negatives) > 0 else 0.0


    #######    
    ####### ID_S4_EX3 END #######     
    print('precision = ' + str(precision) + ", recall = " + str(recall))   

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)
    

    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

