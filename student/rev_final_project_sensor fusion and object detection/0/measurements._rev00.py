


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
#이 파일은 LiDAR 및 카메라 센서로부터 측정한 데이터를 처리하고, 그 결과를 추적 알고리즘에 사용할 수 있는 형태로 변환하는 작업을 담당함.


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
    def __init__(self, name, calib):
        self.name = name
        if name == 'lidar':
            self.dim_meas = 3  # 측정 차원 (x, y, z)
            self.sens_to_veh = np.matrix(np.identity((4)))  # LiDAR 센서는 단위 행렬로 초기화
        elif name == 'camera':
            self.dim_meas = 2  # 측정 차원 (u, v)
            self.sens_to_veh = calib
        else:
            raise ValueError('Unknown sensor type')
        
        self.fov = [-np.pi / 4, np.pi / 4]  # 시야각 (카메라에 사용)

    def get_H(self, x):
        '''상태 벡터 x에 대한 Jacobian 행렬 H 반환'''
        H = np.matrix(np.zeros((self.dim_meas, params.dim_state)))
        R = self.sens_to_veh[0:3, 0:3]  # 회전 행렬
        T = self.sens_to_veh[0:3, 3]  # 변환 벡터
        
        if self.name == 'lidar':
            H[0:3, 0:3] = R  # LiDAR의 경우
        elif self.name == 'camera':
            # Jacobian 행렬 계산 중 분모가 0이 되는지 확인
            if R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0] == 0:
                raise ValueError('Jacobian not defined for this x!')
            else:
                H[0, 0] = self.f_i * (-R[1, 0] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                                     + R[0, 0] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1])
                                     / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
                H[1, 0] = self.f_j * (-R[2, 0] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                                     + R[0, 0] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2])
                                     / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
                H[0, 1] = self.f_i * (-R[1, 1] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                                     + R[0, 1] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1])
                                     / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
                H[1, 1] = self.f_j * (-R[2, 1] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                                     + R[0, 1] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2])
                                     / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
                H[0, 2] = self.f_i * (-R[1, 2] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                                     + R[0, 2] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1])
                                     / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
                H[1, 2] = self.f_j * (-R[2, 2] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                                     + R[0, 2] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2])
                                     / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
        return H
    
    def in_fov(self, x):
        '''객체가 카메라의 시야에 있는지 확인'''
        pos_veh = np.ones((4, 1))
        pos_veh[0:3] = x[0:3]
        pos_sens = np.dot(self.sens_to_veh, pos_veh)  # 센서 좌표계로 변환
        angle = np.arctan2(pos_sens[1], pos_sens[0])
        visible = self.fov[0] <= angle <= self.fov[1]
        return visible

    def get_hx(self, x):
        '''비선형 카메라 측정 모델 h(x) 반환'''
        if self.name == 'lidar':
            return x[0:self.dim_meas]
        elif self.name == 'camera':
            px, py, pz = x[0], x[1], x[2]
            denom = np.sqrt(px**2 + py**2)
            if denom == 0:  # 분모가 0이 되는 경우 방지
                raise ValueError("Division by zero in h(x) calculation")
            hx = np.array([[np.arctan2(py, px)], [pz / denom]])
            return hx



################### 
        
class Measurement:
    '''Measurement class including measurement values, covariance, timestamp, sensor'''
    def __init__(self, num_frame, z, sensor):
        # create measurement object
        self.t = (num_frame - 1) * params.dt # time
        if sensor.name == 'lidar':
            sigma_lidar_x = params.sigma_lidar_x # load params
            sigma_lidar_y = params.sigma_lidar_y
            sigma_lidar_z = params.sigma_lidar_z
            self.z = np.zeros((sensor.dim_meas,1)) # measurement vector
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.z[2] = z[2]
            self.sensor = sensor # sensor that generated this measurement
            self.R = np.matrix([[sigma_lidar_x**2, 0, 0], # measurement noise covariance matrix
                                [0, sigma_lidar_y**2, 0], 
                                [0, 0, sigma_lidar_z**2]])
            
            self.width = z[4]
            self.length = z[5]
            self.height = z[3]
            self.yaw = z[6]
        elif sensor.name == 'camera':
            
            ############
            # TODO Step 4: initialize camera measurement including z, R, and sensor 
            ############

            sigma_cam_i = params.sigma_cam_i
            sigma_cam_j = params.sigma_cam_j
            
            self.z = np.zeros((sensor.dim_meas,1))
            self.z[0] = z[0]
            self.z[1] = z[1]
            self.sensor = sensor
            self.R = np.matrix([[sigma_cam_i**2 , 0],
                                [0,sigma_cam_j**2]])
            
            self.width = z[2]
            self.length = z[3]
        
            ############
            # END student code
            ############ 
