#이 파일은 LiDAR 및 카메라 센서로부터 측정한 데이터를 처리하고, 그 결과를 추적 알고리즘에 사용할 수 있는 형태로 변환하는 작업을 담당합니다.


# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for sensor and measurement 
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Sensor:
    '''Sensor class including measurement matrix'''
    #__init__(self, name, calib) : 역할: 센서 객체를 초기화하는 생성자입니다.
    def __init__(self, name, calib): 
        #인자: name: 센서 이름 ('lidar' 또는 'camera')
        #인자: calib: 카메라 센서일 때 사용할 외부 및 내부 캘리브레이션 데이터
        self.name = name #변수 : self.name: 센서의 이름
        if name == 'lidar':
            self.dim_meas = 3 #변수 : self.dim_meas: 측정값의 차원 (LiDAR는 3차원, 카메라는 2차원), Lidar센서 사용시, 3차원 공간에서 측정값 수집한다는 의미,측정차원(dim_meas)을 3으로 설정하는 코드
            self.sens_to_veh = np.matrix(np.identity((4))) # transformation sensor to vehicle coordinates equals identity matrix because lidar detections are already in vehicle coordinates
                                                           #변수:self.sens_to_veh: 센서 좌표계를 차량 좌표계로 변환하는 행렬 (LiDAR는 이미 차량 좌표계이므로 단위행렬 사용)
                                                           #LiDAR 센서의 좌표계와 차량 좌표계가 동일하다는 것을 나타내며, 좌표 변환이 필요 없음을 의미
                                                           #이를 위해 4x4 단위 행렬을 사용하여 센서 좌표계를 차량 좌표계로 변환하는 변환 행렬로 설정    
                                                            
             
            
            self.fov = [-np.pi/2, np.pi/2] # angle of field of view in radians 
                                           #np.pi는 **π (파이)**를 나타내며, 약 3.14159의 값 ,np.pi/2는 90도를 의미. 라디안 단위로 계산된 값이므로, 90도 = π/2 라디안.
                                           #self.fov = [-np.pi/2, np.pi/2]: 이 값은 시야각 범위를 라디안 단위로 나타낸 것, [-np.pi/2, np.pi/2]는 좌우 90도씩, 총 180도의 시야각
                                           #-np.pi/2: 좌측 90도 (왼쪽 끝), np.pi/2: 우측 90도 (오른쪽 끝)
                                           #변수: self.fov: 센서의 시야각 (LiDAR와 카메라 모두 각기 다른 시야각을 가짐)
                                           #self.fov = [-np.pi/2, np.pi/2]는 LiDAR 센서의 시야 범위를 좌우 90도씩 설정하여, 총 180도의 시야각을 가지도록 설정한 것. LiDAR 센서가 전방 180도 범위 내에서 물체를 감지할 수 있음을 의미
        
        
        elif name == 'camera':
            self.dim_meas = 2 #카메라에서 측정되는 데이터의 차원, 카메라는 보통 2D 이미지로 정보를 표현하므로, **2차원 좌표 (i, j)**로 표현, i: 가로 좌표 (x축에 해당), j: 세로 좌표 (y축에 해당)
            self.sens_to_veh = np.matrix(calib.extrinsic.transform).reshape(4,4) # transformation sensor to vehicle coordinates
                                                                                 #변수 : self.f_i, self.f_j, self.c_i, self.c_j: 카메라의 초점 거리 및 주점 좌표 (카메라일 때만 사용)
                                                                                 #카메라 센서에서 **차량 좌표계(vehicle coordinates)**로 변환하는 변환 행렬을 정의
                                                                                 #이 변환은 센서에서 얻은 좌표를 차량 좌표계로 변환하는 역할
            self.f_i = calib.intrinsic[0] # focal length i-coordinate 
                                          #calib.extrinsic.transform는 **카메라 센서의 외부 보정 정보 (extrinsics)**를 가져오는 것으로, 이를 4x4 변환 행렬로 **재배열(reshape)**하여, 센서 좌표계 → 차량 좌표계 변환을 수행
                                          #**i축 방향 초점 거리 (focal length)**를 설정 ,calib.intrinsic[0]은 카메라의 보정 데이터에서 초점 거리를 가져오며, 이는 i축 방향으로의 초점 거리
            self.f_j = calib.intrinsic[1] # focal length j-coordinate, j축 방향 초점 거리를 설정,calib.intrinsic[1]은 카메라의 j축 방향 초점 거리
            self.c_i = calib.intrinsic[2] # principal point i-coordinate ,**카메라 주점 (principal point)**의 i축 좌표를 설정,calib.intrinsic[2]는 i축에서의 주점 좌표
                                          #**주점(principal point)**은 카메라 렌즈를 통해 이미지를 얻을 때, 이미지의 중심을 나타내는 좌표 
            self.c_j = calib.intrinsic[3] # principal point j-coordinate, 카메라 주점의 j축 좌표를 설정,calib.intrinsic[3]는 j축에서의 주점 좌표

            self.fov = [-0.35, 0.35] # angle of field of view in radians, inaccurate boundary region was removed,카메라의 **시야각(Field of View, FOV)**을 설정
                                     #이 값은 라디안 단위로 표현된 시야 범위이며, 카메라가 감지할 수 있는 수평 시야각 범위가 약 40도 (0.35 라디안 = 약 20도)임을 나타냄
                                     #[-0.35, 0.35]: 카메라가 감지할 수 있는 수평 범위가 좌우로 각각 20도씩, 총 40도
            #이 코드는 카메라의 **초점 거리(f_i, f_j)**와 주점 좌표(c_i, c_j) 등을 설정하여, 카메라의 이미지 좌표계에서 차량 좌표계로 변환하는 과정에서 필요한 보정 정보를 초기화하고, 카메라의 시야각을 설정하는 부분임                         
            
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh) # transformation vehicle to sensor coordinates
                                                           #self.veh_to_sens: 차량 좌표계에서 센서 좌표계로 변환하는 역행렬.
                                                           #self.sens_to_veh: 센서에서 차량 좌표계로 변환하는 행렬.
                                                           #elf.veh_to_sens = np.linalg.inv(self.sens_to_veh)는 차량 좌표계에서 센서 좌표계로 변환하는 변환 행렬을 설정하는 부분
                                                            #*self.sens_to_veh**는 센서 좌표계에서 차량 좌표계로 변환하는 변환 행렬입니다. 즉, 센서에서 얻은 데이터(예: 라이다나 카메라의 측정 데이터)를 차량 좌표계로 변환하기 위한 행렬
                                                            #**np.linalg.inv(self.sens_to_veh)**는 변환 행렬의 역행렬을 계산하는 함수, 역행렬을 구하는 이유는, 차량 좌표계에서 센서 좌표계로 변환하려면, 센서 → 차량 변환 행렬의 역행렬이 필요하기 때문
                                                            #차량 좌표계에서 센서 좌표계로 변환하려면 반대 방향의 변환을 해야 하므로, 역행렬을 사용해 그 변환을 수행
#예시:카메라나 라이다 같은 센서가 차량에 장착되어 있다면, 센서는 센서 좌표계에서 데이터를 기록. 그러나 이 데이터를 차량의 차량 좌표계에서 사용하려면 좌표 변환이 필요
#**self.sens_to_veh**는 센서의 데이터를 차량 좌표계로 변환하고, **self.veh_to_sens**는 다시 그 데이터를 센서 좌표계로 변환할 수 있도록 함

                                                            
    def in_fov(self, x):# 카메라 센서에 대해 시야각 내에 있는지 확인하는 기능을 구현
        #인자 : x -->객체의 차량 좌표계에서의 위치
        # check if an object x can be seen by this sensor
        ############
        # TODO Step 4: implement a function that returns True if x lies in the sensor's field of view, 
        # otherwise False.
        ############
        #카메라의 시야내에 있는지 판단하는 기능 , lidar는 기본적으로 차량좌표계 사용하므로 별도처리 필요없음
        if self.name == 'camera':
            pos_veh = np.ones((4, 1)) #변수, pos_veh: 차량 좌표계에서의 객체 위치
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens * pos_veh #변수 , pos_sens: 센서 좌표계에서의 객체 위치
            alpha = np.arctan2(pos_sens[1], pos_sens[0]) #변수, alpha: 객체가 센서 시야 내에 있는지 판단하기 위한 각도
            if alpha > self.fov[0] and alpha < self.fov[1]:
                return True
            else:
                return False
        return True  # lidar는 기본적으로 차량 좌표계를 사용하므로 항상 True




        #return True
        ############
        # END student code
        ############ 
             
    def get_hx(self, x): #카메라 센서의 비선형 변환을 구현, 물체의 위치를 카메라 좌표계에서 변환하고 이미지 좌표계로 투영하는 작업수행
                         # 또는 측정기대값을 비선형 함수로 계산하는 함수
                         #인자, x:차량 좌표계에서의 객체 상태 벡터
        # calculate nonlinear measurement expectation value h(x)   
        if self.name == 'lidar':
            pos_veh = np.ones((4, 1)) # homogeneous coordinates
            #변수,pos_veh: 차량 좌표계에서의 객체 위치
            pos_veh[0:3] = x[0:3] 
            pos_sens = self.veh_to_sens*pos_veh # transform from vehicle to lidar coordinates
            #변수,pos_sens: 센서 좌표계에서의 객체 위치
            return pos_sens[0:3]
        #물체의 좌표를 카메라 좌표계로 변환 후, 이미지 좌표계로 투영, 예외상황(0)으로 나누는 경우도 처리
        elif self.name == 'camera':
            # 카메라 좌표계로 변환 후 이미지 좌표계로 투영
            pos_veh = np.ones((4, 1))
            pos_veh[0:3] = x[0:3]
            pos_sens = self.veh_to_sens * pos_veh
            if pos_sens[0] == 0:  # x축 값이 0인 경우 예외 처리
                raise ZeroDivisionError('Division by zero in camera projection')
            i = self.f_i * pos_sens[0] / pos_sens[2] + self.c_i
            j = self.f_j * pos_sens[1] / pos_sens[2] + self.c_j
            return np.array([i, j])
            #변수 ,i, j: 카메라 이미지 좌표계로 투영된 객체의 위치

            ############
            # TODO Step 4: implement nonlinear camera measurement function h:
            # - transform position estimate from vehicle to camera coordinates
            # - project from camera to image coordinates
            # - make sure to not divide by zero, raise an error if needed
            # - return h(x)
            ############

            #pass
            ############
            # END student code
            ############ 
        
    def get_H(self, x): #get_H(self, x) ,역할: 상태 벡터에 대한 Jacobian 행렬을 계산하는 함수
        # calculate Jacobian H at current x from h(x)
        #인자,x: 상태 벡터
        #변수,H: Jacobian 행렬
        #변수,  R, T: 센서에서 차량 좌표계로 변환하기 위한 회전 및 변환 행렬
        H = np.matrix(np.zeros((self.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3] # rotation
        T = self.veh_to_sens[0:3, 3] # translation
        if self.name == 'lidar':
            H[0:3, 0:3] = R
        elif self.name == 'camera':
            # check and print error message if dividing by zero
            if R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0] == 0: 
                raise NameError('Jacobian not defined for this x!')
            else:
                H[0,0] = self.f_i * (-R[1,0] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,0] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[1,0] = self.f_j * (-R[2,0] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,0] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[0,1] = self.f_i * (-R[1,1] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,1] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[1,1] = self.f_j * (-R[2,1] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,1] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[0,2] = self.f_i * (-R[1,2] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,2] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
                H[1,2] = self.f_j * (-R[2,2] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                    + R[0,2] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                        / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
        return H   
        
    def generate_measurement(self, num_frame, z, meas_list): 
        #카메라 센서 측정값을 추가하는 작업수행
        #역할: 새로운 측정값을 생성하고 리스트에 추가하는 함수
        #인자: num_frame: 현재 프레임 번호
        #인자: z: 센서에서 측정된 데이터
        #인자 : meas_list: 측정값 리스트


        # generate new measurement from this sensor and add to measurement list
        ############
        # TODO Step 4: remove restriction to lidar in order to include camera as well
        ############
        #카메라 센서로부터 측정값 생성하는 코드 추가, 이제 lidar와 camera모두 처리할 수 있음.
        
        if self.name == 'lidar':
            meas = Measurement(num_frame, z, self)
            meas_list.append(meas)
        elif self.name == 'camera':
            meas = Measurement(num_frame, z, self)
            meas_list.append(meas)

        return meas_list
        
        ############
        # END student code
        ############ 
        
        
################### 
        
class Measurement:
    '''Measurement class including measurement values, covariance, timestamp, sensor'''
    def __init__(self, num_frame, z, sensor):
        #__init__(self, num_frame, z, sensor) --> 역할: 측정값 객체를 초기화하는 생성자
        #인자, num_frame: 프레임번호
        #인자, z: 센서에서 측정된 데이터
        #인자, sensor: 측정값을 생성한 센서 객체

        #변수
        #self.t: 측정이 이루어진 시간
        #self.sensor: 측정값을 생성한 센서
        #self.z: 측정값 벡터
        #self.R: 측정 노이즈 공분산 행렬
        #self.width, self.length, self.height, self.yaw: LiDAR에서 측정한 객체의 너비, 길이, 높이, 요(yaw) 값

        '''
        sigma_lidar_x, sigma_lidar_y, sigma_lidar_z는 LiDAR 센서에서 측정된 값의 **측정 노이즈(Measurement Noise)**를 나타내는 표준편차입니다. 
        즉, LiDAR 센서가 물체의 위치를 측정할 때, 
        각각의 축(x, y, z)에서 발생할 수 있는 측정 오류를 설명합니다. 
        일반적으로, 센서로부터 측정된 값은 완전히 정확하지 않기 때문에, 그 불확실성을 고려하여 노이즈 값을 설정합니다.
        '''

        # create measurement object
        self.t = (num_frame - 1) * params.dt # time
        self.sensor = sensor # sensor that generated this measurement
        
        if sensor.name == 'lidar':
            sigma_lidar_x = params.sigma_lidar_x # load params #sigma_lidar_x: x축 방향의 측정 오차를 나타내는 표준편차
                                                 #params.sigma_lidar_x는 어디선가 이미 설정된 값이고, 이 값을 전역 설정 파일이나 파라미터 파일(params)에서 가져오고 있는 것
                                                 #params.sigma_lidar_x: params라는 설정 파일이나 모듈에서 sigma_lidar_x라는 이름으로 정의된 변수.
                                                 #sigma_lidar_x = params.sigma_lidar_x: 코드에서 params에 있는 sigma_lidar_x 값을 현재 사용하려는 변수 sigma_lidar_x에 할당하여 사용하는 것.
                                                 #결국, 이 코드는 LiDAR 센서의 x축 방향 측정 노이즈 표준편차 값을 params라는 설정 파일에서 가져와서 해당 값을 사용한다는 뜻#
                                                 # params 파일에서 각종 센서 파라미터들을 관리하고 있고, 이를 통해 코드에서 쉽게 해당 파라미터를 참조할 수 있게 만든 것

            sigma_lidar_y = params.sigma_lidar_y #sigma_lidar_y: y축 방향의 측정 오차를 나타내는 표준편차
            sigma_lidar_z = params.sigma_lidar_z #sigma_lidar_z: z축 방향의 측정 오차를 나타내는 표준편차
            self.z = np.zeros((sensor.dim_meas,1)) # measurement vector
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.z[2] = z[2]
            self.R = np.matrix([[sigma_lidar_x**2, 0, 0], # measurement noise covariance matrix
                                [0, sigma_lidar_y**2, 0], 
                                [0, 0, sigma_lidar_z**2]])
            
            self.width = z[4]
            self.length = z[5]
            self.height = z[3]
            self.yaw = z[6]
        elif sensor.name == 'camera':
                    
            ############
            # TODO Step 4: initialize camera measurement including z and R 
            ############
            #측정 벡터(z): 카메라에서 측정한 데이터를 기반으로 z 값을 설정.
            #측정 노이즈 공분산 행렬(R): 카메라 측정에 대한 노이즈 값을 반영한 공분산 행렬을 설정.
            sigma_camera_i = params.sigma_camera_i  # i축 카메라 측정 노이즈
            sigma_camera_j = params.sigma_camera_j  # j축 카메라 측정 노이즈
            self.z = np.zeros((sensor.dim_meas, 1))  # 측정 벡터 초기화
            self.z[0] = z[0]  # i축 측정값
            self.z[1] = z[1]  # j축 측정값
            self.R = np.matrix([[sigma_camera_i**2, 0],  # 측정 노이즈 공분산 행렬
                        [0, sigma_camera_j**2]])

            #pass
        
            ############
            # END student code
            ############ 