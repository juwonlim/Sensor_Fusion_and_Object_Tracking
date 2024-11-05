
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


# 이 파일은 LiDAR 및 카메라 센서로부터 측정한 데이터를 처리하고, 그 결과를 추적 알고리즘에 사용할 수 있는 형태로 변환하는 작업을 담당함.

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
    # __init__(self, name, calib): 역할: 센서 객체를 초기화하는 생성자
    def __init__(self, name, calib): 
        # 인자: name: 센서 이름 ('lidar' 또는 'camera')
        # 인자: calib: 카메라 센서일 때 사용할 외부 및 내부 캘리브레이션 데이터
        self.name = name  # 변수: 센서의 이름
        if name == 'lidar':
            self.dim_meas = 3  # LiDAR는 3차원 측정값 수집
            self.sens_to_veh = np.matrix(np.identity((4)))  # LiDAR 센서는 단위 행렬로 초기화
        elif name == 'camera':
            self.dim_meas = 2  # 카메라는 2차원 (i, j) 측정
            self.sens_to_veh = calib  # 외부 보정 데이터 적용
        else:
            raise ValueError('Unknown sensor type')
        
        self.fov = [-np.pi / 4, np.pi / 4]  # 카메라 시야각 설정

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
        pos_veh[0:3] = x[0:3]  # 차량 좌표계에서의 객체 위치
        pos_sens = np.dot(self.sens_to_veh, pos_veh)  # 센서 좌표계로 변환
        angle = np.arctan2(pos_sens[1], pos_sens[0])  # 각도 계산
        visible = self.fov[0] <= angle <= self.fov[1]  # 시야각 내 여부 확인
        return visible

    def get_hx(self, x):
        '''비선형 카메라 측정 모델 h(x) 반환'''
        if self.name == 'lidar':
            return x[0:self.dim_meas]  # LiDAR는 직접 반환
        elif
