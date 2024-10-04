# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Loop over all frames in a Waymo Open Dataset file,
#                        detect and track objects and visualize results
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

##################
## Imports

## general package imports
import os
import sys
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import copy

## Add current working directory to path
sys.path.append(os.getcwd())


#https://www.youtube.com/watch?v=JLKDm3J4Ojs&t=2s
#상단에서 동일한 import파일을 찾을 수 있는데,  전에 언급했던 OpenCV, System import 및 기타의 것들임. 
#웨이모 공개 데이터 세트 프레임 처리를 위해 사용하는 웨이모 공개 데이터 세트 리더를 사용함.(0:57 / 5:32)

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2

#일부 신규 IMPORT파일 추가 
#여기 솔루션은 여러분이 가진 것과 다름에 유의해야함 (1:10 / 5:32)
## 3d object detection
import student.objdet_pcl as pcl  #student.objdet_PCL import가 있는데 여러분은 프로그램이 정상 동작하도록 솔루션코드를 채워넣어야 함.그렇지만 이를 실행하여 여러분에게 이것이 어떤지 보여드리려고 하기 때문에
import student.objdet_detect as det  #여기서 import한 솔루션을 사용하겠음.(1:26 / 5:32)
import student.objdet_eval as eval

import misc.objdet_tools as tools 
from misc.helpers import save_object_to_file, load_object_from_file, make_exec_list


#또한 추적을 위해 일부 파일을 import하는데, 이는 다시 [inaudible] 부분이지만, 일단 코드를 흛은 후 학생을 위해 이를 바꾸는 것과 같은 방식으로 [inaudible]은 또한 이를 학생에게 바꿀 것임.
## Tracking
from student.filter import Filter
from student.trackmanagement import Trackmanagement
from student.association import Association
from student.measurements import Sensor, Measurement
from misc.evaluation import plot_tracks, plot_rmse, make_movie
import misc.params as params 
 
##################
## Set parameters and perform initializations

#특정 웨이모파일을 선택하기 위한 불러오기 기능이 있는데, 이는 이전과 같음. 시퀀스1~3, 또한 여기서 처리할 프레임을 선택할 수 있음.
## Select Waymo Open Dataset file and frame numbers
data_filename = 'training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord' # Sequence 1
# data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
# data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 3
show_only_frames = [0, 200] # show only frames in interval for debugging

## Prepare Waymo Open Dataset file for loading
data_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', data_filename) # adjustable path in case this script is called from another working directory
results_fullpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results')
datafile = WaymoDataFileReader(data_fullpath)
datafile_iter = iter(datafile)  # initialize dataset iterator


#이제 기본 루프와 데이터 세트에 대한 루프 사이의 첫 번째 차이점임.(1:59 / 5:32)
#여기서 객체 감지를 초기화 할 것임.
#실질적으로 사용할 모델을 선택 할 수 있는 부분임. 기 설정 모델은 다크넷이 될 것임. 여러분의 작업에서 수행해야할 다른 모델 혹은 여러분이 구현하거나 여기 이 코드에 통합해야 할 다른 작업은
# FPM resnet임 (2:22 / 5:32) resnet을 공진이라고 번역하고 있음(ai의 오류)
#최종 프로젝트 수행시 사용해야 할 모델이기도 함.그렇지만 지금은 다크넷임.다크넷은 컴플렉스 옐로우와 같음. 이는 그저 모델이름임.
#그러나 다크넷을 선택하면 여기에서 컴플렉스 옐로우 감지 네트워크를 구동하게 됨.(2:40 / 5:32)
#모든 것이 원활하고 정확하게 작동하는지 확인하려면 실제 객체로써 웨이모 정답 값 레이블 사용 여부를 결정할 수 있음.


## Initialize object detection
configs_det = det.load_configs(model_name='fpn_resnet') # options are 'darknet', 'fpn_resnet'
model_det = det.create_model(configs_det)

#이는 완벽한 객체와 완전 감지로 이루어진 완벽한 세계가 될 것임.예를 들어 성능평가를 수행하는 경우와 같이 무언가를 시험하고자 할 때에는 이를 false로 설정하며(3:00 / 5:32)
#만약 이를 true로 설정하면 여러분이 구현하는 모든 것의 성능평가는 틀림없이 100%의 결과를 반환할 것임. 이는 순전히 여러분의 편의를 위한 디버깅 용도 일 뿐임 (3:13 / 5:32)
configs_det.use_labels_as_objects = False # True = use groundtruth labels as objects, False = use model-based detection

#또한 안전결과 플래그가 여기 있음(3:17 / 5:32)
#configs_det.save_results = False #save results to file (based on data_filename) #이를 true로 설정하면,시스템은 실제로 바이너리를 미리 연산하여 폴더에 넣어둠. 곧 보여드리겠음.
#그렇지만 여기 이것을 true로 설정하면 지금까지 구현한 모든 단계는 미리 계산디어 파일에 저장됨. 이는 문제 해결에 편리한 방법인데 첫번째 세가지 연습문제를 해결하고 네번째 연습 문제를 푸는 
#경우에, 연습 문제 1,2,3을 처리하는데는 시간이 걸리기 때문에, 이러한 중간 단계를 미리 계산한 다음 ,결과를 불러와서 4단계,연습문제 4번의 입력으로 사용함.
#이리하면 시간이 절약됨.(3:55 / 5:32) 이것이 그렇게 하기위한 스위치임.




## Uncomment this setting to restrict the y-range in the final project
# configs_det.lim_y = [-25, 25] 



#추적 초기화가 [inaudible]이 언급하려는 것임.여기 이것은 매우 중요함. 

## Initialize tracking
KF = Filter() # set up Kalman filter 
association = Association() # init data association
manager = Trackmanagement() # init track manager
lidar = None # init lidar sensor object
camera = None # init camera sensor object
np.random.seed(10) # make random values predictable


#선택,실행 및 시각화 임 (4:07 / 5:32) 여기 있는 목록, 출구감지, 추적 및 시각화는 여러분의 코드에 따라 어떤 부분이 실제로 연산될지 혹은 파일에서 미리 불러올 지 선택할 수 있도록
#하기위한 것임. 

## Selective execution and visualization
#exec_detection = ['bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'] # options are 'bev_from_pcl', 'detect_objects', 'validate_object_labels', 'measure_detection_performance'; options not in the list will be loaded from file
exec_detection = [] #여기서 모든 문자열을 지우고 빈 목록으로 만듭시다. (4:24 / 5:32) 여러분의 편의를 위해 여기서 옵션을 사용할 수 있음. 예를 들어 포인트 클라우드에서 조감도 연산하려면
                    #여기 문자열 뒤를 유의해야 함. (bev_from_pcl) 이 함수를 구동하려하면, 실제로는 그저 이것을 여기로 복사하고 여기에('[]') 두면 이 함수는 작동하게 됨 (4:49 / 5:32)
                    #지금은 물론 시스템은 솔루션 코드를 사용할 것임. 그런데 일단 이 패키지 import를 학생용으로 변경하면 시스템은 여기 이 문자열에 연관된 함수를 호출하고 여러분이 실제 구현을 완료하리라 예상할 것임.(5:04 / 5:32)
                    #문자열 관련하여 여기 연습 문제에서 작업하고 있다면 여기에 넣고, 만약 제거하는 경우에는 시스템은 여러분에게 제공해드린 기 저장된 바이너리를 사용할 것이므로 결과가 어떻게 보일지 살펴볼 수 있음.
                    #여기가 모두 공란이면,시스템은 기 연산된 바이너리가 적합한 장소에 있으며,이 바이너리에 대해 올바른 시퀀스를 선택했다고 가정하고 기 연산된 바이너리를 사용할 것임.
exec_tracking = [] # options are 'perform_tracking'
#exec_visualization = [] # options are 'show_range_image', 'show_bev', 'show_pcl', 'show_labels_in_image', 'show_objects_and_labels_in_bev', 'show_objects_in_bev_labels_in_camera', 'show_tracks', 'show_detection_performance', 'make_tracking_movie'
exec_visualization = ['show_objects_and_labels_in_bev'] #https://www.youtube.com/watch?v=Ds1im7MCGxs 시각화 (0:48 / 3:41) # 시각화 강의 : 여기에서는 이 레포트를 여기로 보냈을때 제공한 기 연산 결과를 보게 됨(1:05 / 3:41)
                                                        #시각화강의 : 이를 수행하기 위해 저는 항상 디버그 옵션을 사용하는데, 보시듯이 시각화 창이 있음(1:14 / 3:41)-강의에서 보여주는 영상 참조
                                                        #시각화 강의 : 이 창에서는 후반 클라우드 포인트의 조감도를 볼 수 있는데, 여기서 보이는 다양한 색상에 대해서는 개별수업에서 자세히 설명함(1:27 / 3:41)
                                                        #여기서부터는 화면을 보아야해서 ppt에 필기함 ppt918페이지 

exec_list = make_exec_list(exec_detection, exec_tracking, exec_visualization)
##https://www.youtube.com/watch?v=Ds1im7MCGxs 시각화 (위의) 이 목록(exec_detection과 exec_tracking)은 비어 있으므로, 시스템은 단순히 기 연산 파일에서 모든 중간단계에 대한 정보를 불러움(1:00 / 3:41)




#https://www.youtube.com/watch?v=5KmGaRf4T-E  이 줄 아래부터는 이 영상이 출처임
vis_pause_time = 0 # set pause time between frames in ms (0 = stop between frames until key is pressed) #시스템이 두 프레임을 처리하는 사이에 대기하게 됨.  
                   #0은 여러분이 키보드의 키를 누를때까지 대기함을 의미하는데, 스페이스 이건 엔터이건 관계없음.어떻게 되는지 볼 수 있도록 여기에 남겨 두겠음.


##################
#다시 메인루프~
## Perform detection & tracking over all selected frames

#이 무한 루프는 마지막 프레임에 도달하고 시스템이 한 단계 더 진행하려고 시도 했을 때에만 중단되어 이 반복 중지 부분이 basicloop.pi에서와 같이 호출됨.
cnt_frame = 0 
all_labels = []
det_performance_all = [] 
np.random.seed(0) # make random values predictable
if 'show_tracks' in exec_list:    
    fig, (ax2, ax) = plt.subplots(1,2) # init track plot

while True:
    try:
        ## Get next frame from Waymo dataset  #1. 여기서부터
        frame = next(datafile_iter) #프레임은 현재 프레임 카운트,프레인 번호q,프레임번호 2등등의 모든 정보를 포함함(#https://www.youtube.com/watch?v=5KmGaRf4T-E : 0:42 / 5:10) 여기가 모든 정보가 있는 곳임.
        if cnt_frame < show_only_frames[0]:
            cnt_frame = cnt_frame + 1
            continue
        elif cnt_frame > show_only_frames[1]:
            print('reached end of selected frames')
            break
        
        print('------------------------------')
        print('processing frame #' + str(cnt_frame)) #1.여기까지 다른 웨이모 프레임을 처리하게 되는데,이전과 같은 코드임.

        #################################

        #그러면 실제 3d객체감지로 가보죵 (https://www.youtube.com/watch?v=5KmGaRf4T-E : 0:49 / 5:10)
        ## Perform 3D object detection

        ## Extract calibration data and front camera image from frame (첫번째 단계는 현재 프레임에서 라이다와 전면 카메라의 보정 데이터를 추출하는 것임, 따라서 라이다 이름과 카메라 이름이 필요함 )
        lidar_name = dataset_pb2.LaserName.TOP #라이다 이름 , 웨이모 차량 지붕의 상단에 있는 상단 라이다 센서 이름을 불러옵니당. ( 1:41 / 5:10 )
        camera_name = dataset_pb2.CameraName.FRONT #카메라 이름, (강의에서는)dataset_p2아래에 오류 표시줄이 있는 이유는 비주얼 코드 인텔리센스 플러그인 입력경로 및 콘텐츠가 거기 실제로 있는지와 관련이 있음.
                                                   # 왜 이런 오류가 발생하는지 모르지만 (1:19 / 5:10) 여러분의 시스템에서도 발생할지도 모름. 이 오류는 신경쓰지 마셈. LaserName클래스에 TOP멤버가 없다고 함.
                                                   # 사실 TOP멤버가 있으므로 그냥 무시하셈.
                                                   # 그리고 전면카메라를 불러오는데, 여기 이것임(1:45 / 5:10)   
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name) #그 다음에는 단순 웨이모리더의 웨이모 유틸리티에 프레임의 실제 보정 정보를 전달하여 라이다 보정을 불러옴. 
                                                                                          #또한 시스템에 어떤 라이다 센서를 불러올지 알림. 카메라 보정도 동일함 (2:00 / 5:10)      
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)

        #이미지 불러오기 문자열이 실행목록에 로드되면 시스템이 카메라 이미지도 불러오도록 할 수 있음. 
        if 'load_image' in exec_list:
            image = tools.extract_front_camera_image(frame) 
        #이미지 배치에서 실제 라이다 포인트 클라우드를 불러오는 첫 단계를 진행해 보장 (2:15 / 5:10)
        #따라서 여러분이 이 문자열을 실행 목록에 넣으면( if 'pcl_from_rangeimage' in exec_list:)
        #
        ## Compute lidar point-cloud from range image    
        if 'pcl_from_rangeimage' in exec_list: #여기에 있는 이러한 문자열은 예를 들어 추적,객체감지 또는 시각화를 위한 실행 목록에 입력한 문자열임(2:51 / 5:10)
            print('computing point-cloud from lidar range image') 
            lidar_pcl = tools.pcl_from_range_image(frame, lidar_name) #시스템은 pcl_from_range_image 함수 호출을 시도하고 단순 범위 이미지에서 포인트 클라우드를 불러옴 (2:30 / 5:10)
                                                                      
        else:
            print('loading lidar point-cloud from result file') ##이를 수행하지 않으면,시스템은 파일에서 기 연산된 포인트 클라우드를 불러와서 여러분의 편의를 위해 제공한 헬퍼 함수 중 하나인 이 함수로 그렇게 처리함.(2:43 / 5:10)
            lidar_pcl = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', cnt_frame)
            
        ## Compute lidar birds-eye view (bev)
        #이제 조감도, 즉 BV맵 연산에 관련된 두번째 단계로 진행함 (2:57 / 5:10) 사실 여러분이 작업해야할 첫번째 연습문제중 하나임.
        #bev_from_pcl은 함수이름이며,이 함수를 들여다보고 싶다면, 잠시 보여드리면 이것을 찾을 수 있을텐데,우선 계속 진행함. (3:13 / 5:10)
        #다시, 이 문자열(bev_from_pcl)을 실행 목록에 넣지 않으면 시스템은 여러분이 파일에서 불러오기를 원한다고 추정함.(3:20 / 5:10)
        if 'bev_from_pcl' in exec_list: 
            print('computing birds-eye view from lidar pointcloud')
            lidar_bev = pcl.bev_from_pcl(lidar_pcl, configs_det)
        else:
            print('loading birds-eve view from result file')
            lidar_bev = load_object_from_file(results_fullpath, data_filename, 'lidar_bev', cnt_frame)

        #다음은 그 다음 단계로 3D객체 감지임. 다시 여러분이 작업할 함수가 있는데(아래의 convert_labels_into_objects )사실 이것으로 작업할 필요는 없음(3:30 / 5:10)
        #이는 여러분을 위해 저희가 작성한,편리한 기 작성 함수임. 여러분이 실제로 할 일은 , 전에 언급한 것처럼 객체에 객체 레이블을 사용하기로 결정한 경우에 쓰이는 것임 (3:41 / 5:10)
        #그러나 사용하지 않기로 결정한 경우에는, 알아봐야 하는 문자열임. 여러분 스스로 실제 객체를 감지하기로 결정한 경우에는 (밑의 else문)
        ## 3D object detection
        if (configs_det.use_labels_as_objects==True): #이게 객체레이블인가봄(3:41 / 5:10 구간에서 하이라이트 해주는 곳)
            print('using groundtruth labels as objects')
            detections = tools.convert_labels_into_objects(frame.laser_labels, configs_det)
        else:
            if 'detect_objects' in exec_list: #여러분 스스로 실제 객체를 감지하기로 결정한 경우에는 여기 이 함수, det.detect_objects를 작업해야 함(4:00 / 5:10)
                print('detecting objects in lidar pointcloud')   
                detections = det.detect_objects(lidar_bev, model_det, configs_det) #이것이 학생들이 구현해야할 코드부분이며,여기에서 이 함수를 설명대로 구현해야 함(4:08 / 5:10)
            else:
                print('loading detected objects from result file')
                # load different data for final project vs. mid-term project
                if 'perform_tracking' in exec_list:
                    detections = load_object_from_file(results_fullpath, data_filename, 'detections', cnt_frame)
                else:
                    detections = load_object_from_file(results_fullpath, data_filename, 'detections_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)


        #그러면 객체 레이블 검증부분으로 감. 강의 중 하나에서 레이블 검증에 대해 잠시 다룰 것이기 땜시,여기서는 웨이모 데이터세트에서 미리 불러온 각각의 정답 값 레이블의 잘못된 레이블 플래그와
        #감지 성능 평가 시 이를 사용해야만 하는지 여부라는 측면에서의 관련성에 관한 정보 외에는 반복하지 않겠음 (밑의 valid_label_flags를 하이라이트함)(4:34 / 5:10)
        ## Validate object labels
        if 'validate_object_labels' in exec_list:
            print("validating object labels")
            valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects==True else 10)
        else:
            print('loading object labels and validation from result file')
            valid_label_flags = load_object_from_file(results_fullpath, data_filename, 'valid_labels', cnt_frame)            


        #그럼 성능평가 부분으로 넘어가겠음. 여기서는 모든 프레임 각각에 대해 정탐(?)(true positives,true nagatives)등의 측면에서 일부 초기 성능 측정값을 수집함(4:47 / 5:10)
        ## Performance evaluation for object detection
        if 'measure_detection_performance' in exec_list:
            print('measuring detection performance')
            det_performance = eval.measure_detection_performance(detections, frame.laser_labels, valid_label_flags, configs_det.min_iou) 
            #(윗줄 measure_detection_perfornamce하이라이트)여기 이 구조내에 이러한 정보를 누적하는데 (4:50 / 5:10)   
        else:
            print('loading detection performance measures from file')
            # load different data for final project vs. mid-term project
            if 'perform_tracking' in exec_list:
                det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance', cnt_frame)
            else:
                det_performance = load_object_from_file(results_fullpath, data_filename, 'det_performance_' + configs_det.arch + '_' + str(configs_det.conf_thresh), cnt_frame)   

        det_performance_all.append(det_performance) # store all evaluation results in a list for performance assessment at the end
        #(윗줄)여기 이 구조내에 이러한 정보를 누적하는데,단순히 매 프레임에 추가하고 성능 측정값을 여기 이 구조내에 추가함. 그리고 모든 프레임이 처리된 루프마지막에 여기 이 누적구조를 사용하여(아래주석연결)
        #(윗주석에서 연결) 전체 성능 평가를 수행함 (5:09 / 5:10)



        #https://www.youtube.com/watch?v=Ds1im7MCGxs&t=1s
        #다음으로 시각화 부분으로 이동하는데, 여기서는 객체감지인데, 여기에서 여러분이 구현한 다양한 단계의 결과물을 간단히 살펴보고 어떻게 보이는지 확인할 수 있음(0:12 / 3:41)
        
        ## Visualization for object detection
        if 'show_range_image' in exec_list:
            img_range = pcl.show_range_image(frame, lidar_name)
            img_range = img_range.astype(np.uint8)
            cv2.imshow('range_image', img_range)
            cv2.waitKey(vis_pause_time)

        if 'show_pcl' in exec_list:
            pcl.show_pcl(lidar_pcl)

        if 'show_bev' in exec_list:
            tools.show_bev(lidar_bev, configs_det)  
            cv2.waitKey(vis_pause_time)          

        if 'show_labels_in_image' in exec_list:
            img_labels = tools.project_labels_into_camera(camera_calibration, image, frame.laser_labels, valid_label_flags, 0.5)
            cv2.imshow('img_labels', img_labels)
            cv2.waitKey(vis_pause_time)


        #이제 처음으로 코드실행, 객체 및 레이블 조감도로 보여드리겠음.딥러닝 프레임워크가 감지한 실제 객체와 더 많은 데이터 세트와 같이 제공된 정답값 레이블을 시각화하려면 그저 이 코드를 
        #여기에 복사하고,시각화 부분의 실행목록까지 스크롤하여(0:39 / 3:41)(이때 )selective execution and visualizaton부분까지 올라가서 보여줌(0:48 / 3:41) exec_visualization = ['show_objects_and_labels_in_bev'] 

        if 'show_objects_and_labels_in_bev' in exec_list:
            tools.show_objects_labels_in_bev(detections, frame.laser_labels, lidar_bev, configs_det)
            cv2.waitKey(vis_pause_time)         

        if 'show_objects_in_bev_labels_in_camera' in exec_list:
            tools.show_objects_in_bev_labels_in_camera(detections, lidar_bev, image, frame.laser_labels, valid_label_flags, camera_calibration, configs_det)
            cv2.waitKey(vis_pause_time)               


        #################################
        ## Perform tracking
        if 'perform_tracking' in exec_list:
            # set up sensor objects
            if lidar is None:
                lidar = Sensor('lidar', lidar_calibration)
            if camera is None:
                camera = Sensor('camera', camera_calibration)
            
            # preprocess lidar detections
            meas_list_lidar = []
            for detection in detections:
                # check if measurement lies inside specified range
                if detection[1] > configs_det.lim_x[0] and detection[1] < configs_det.lim_x[1] and detection[2] > configs_det.lim_y[0] and detection[2] < configs_det.lim_y[1]:
                    meas_list_lidar = lidar.generate_measurement(cnt_frame, detection[1:], meas_list_lidar)

            # preprocess camera detections
            meas_list_cam = []
            for label in frame.camera_labels[0].labels:
                if(label.type == label_pb2.Label.Type.TYPE_VEHICLE):
                
                    box = label.box
                    # use camera labels as measurements and add some random noise
                    z = [box.center_x, box.center_y, box.width, box.length]
                    z[0] = z[0] + np.random.normal(0, params.sigma_cam_i) 
                    z[1] = z[1] + np.random.normal(0, params.sigma_cam_j)
                    meas_list_cam = camera.generate_measurement(cnt_frame, z, meas_list_cam)
            
            # Kalman prediction
            for track in manager.track_list:
                print('predict track', track.id)
                KF.predict(track)
                track.set_t((cnt_frame - 1)*0.1) # save next timestamp
                
            # associate all lidar measurements to all tracks
            association.associate_and_update(manager, meas_list_lidar, KF)
            
            # associate all camera measurements to all tracks
            association.associate_and_update(manager, meas_list_cam, KF)
            
            # save results for evaluation
            result_dict = {}
            for track in manager.track_list:
                result_dict[track.id] = track
            manager.result_list.append(copy.deepcopy(result_dict))
            label_list = [frame.laser_labels, valid_label_flags]
            all_labels.append(label_list)
            
            # visualization
            if 'show_tracks' in exec_list:
                fig, ax, ax2 = plot_tracks(fig, ax, ax2, manager.track_list, meas_list_lidar, frame.laser_labels, 
                                        valid_label_flags, image, camera, configs_det)
                if 'make_tracking_movie' in exec_list:
                    # save track plots to file
                    fname = results_fullpath + '/tracking%03d.png' % cnt_frame
                    print('Saving frame', fname)
                    fig.savefig(fname)

        # increment frame counter
        cnt_frame = cnt_frame + 1    

    except StopIteration:
        # if StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break


#################################
#시각화 :https://www.youtube.com/watch?v=Ds1im7MCGxs

#루프의 끝에 도달하게 됨 (3:16 / 3:41)
#이 문자열을 여기 실행목록(show_detection_performance)에 넣으면,file compute_performance starts에 들어가게 되는데

## Post-processing



## Evaluate object detection performance
if 'show_detection_performance' in exec_list:
    eval.compute_performance_stats(det_performance_all, configs_det) #file compute_performance starts에 들어가게 되는데,det_performance_all 여기에 메인루프를 실행하여
                                                                    #수집한 모든 누적 프레임 관련 성능 측정값을 입력함
                                                                    #기본적으로 거기 모든 것은 현재 여기 이 파일(loop_over_dataset)에 대해 말할 수 있는 모든 것임.이제 구조를 봅시다(시각화 강의 끝)

## Plot RMSE for all tracks
if 'show_tracks' in exec_list:
    plot_rmse(manager, all_labels, configs_det)

## Make movie from tracking results    
if 'make_tracking_movie' in exec_list:
    make_movie(results_fullpath)
