# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

'''
Mid Term: 3D Object Detection 
함수정의

F(): 시스템 행렬을 반환하는 함수. 시스템의 시간 변화를 반영한 상태 변화를 나타냄
Q(): 프로세스 노이즈 공분산 행렬을 반환. 상태의 예측에서 불확실성을 반영
predict(): 상태와 공분산을 다음 시점으로 예측하는 과정.
update(): 측정된 데이터를 사용해 상태와 공분산을 업데이트
gamma(): 측정 잔차(측정값과 예측값의 차이)를 계산.
S(): 잔차의 공분산 행렬을 계산하여 업데이트 과정에 사용
'''


'''
최종 프로젝트 : Sensor Fusion & Object Detection
함수정의 업데이트 :
1.시스템 행렬 F와 프로세스 노이즈 Q: F와 Q 행렬은 mid-term 프로젝트의 구성을 유지하며, 등속 모델 및 프로세스 노이즈 요구를 충족합니다.
2.측정 업데이트: update 함수는 mabhi16의 구조를 따라 gamma와 S를 사용해 칼만 이득을 계산하며 센서 입력의 유연성을 유지합니다.
3.일부 파라미터 수정

'''




# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 

class Filter:
    '''Kalman filter class'''

    def __init__(self):
        # 상태 벡터 초기화 [px, py, vx, vy, ax, ay]
        self.x = np.zeros((6, 1))  

        # 상태 공분산 행렬 초기화
        self.P = np.eye(6)  

        # 프로세스 노이즈 행렬 Q 초기화
        self.Q = np.eye(6)  

        # 측정 노이즈 공분산 행렬 R 초기화
        self.R = np.array([[0.15, 0], [0, 0.15]])  

        # 관측 행렬 H 초기화
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])
        
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        #시스템 행렬 F
        #dt: 시간 변화량을 불러와 등속 모델 행렬 F 생성
        dt = params.dt
        F = np.matrix([[1, 0, dt, 0, 0, 0],
                       [0, 1, 0, dt, 0, 0],
                       [0, 0, 1, 0, dt, 0],
                       [0, 0, 0, 1, 0, dt],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
        return F


       
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        # 프로세스 노이즈 공분산 행렬 Q를 구현
        # q와 dt를 통해 프로세스 노이즈 행렬을 동적으로 계산
        q = params.q
        dt = params.dt
        q1 = ((dt**3) / 3) * q
        q2 = ((dt**2) / 2) * q
        q3 = dt * q
        Q = np.matrix([[q1, 0, 0, q2, 0, 0],
                       [0, q1, 0, 0, q2, 0],
                       [0, 0, q1, 0, 0, q2],
                       [q2, 0, 0, q3, 0, 0],
                       [0, q2, 0, 0, q3, 0],
                       [0, 0, q2, 0, 0, q3]])
        return Q

        
        ############
        # END student code
        ############ 

    def predict(self, track):
         
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        #Opt1
        #상태 x와 공분산 P를 다음 시점으로 예측하여 track에 저장 
        #미래의 상태를 예측한 후 track 객체에 저장된 x(상태벡터)와 p(오차 공분산 행렬)을 (직접)갱신, 그래서 no return 값        
        #F = self.F()
        #track.x = F * track.x  # 상태 예측
        #track.P = F * track.P * F.T + self.Q()  # 공분산 예측
        #pass
        ####################
        #opt2
        #상태 예측 및 오차 공분산 갱신
     
        F = self.F()
        x = F * track.x  # 상태 예측
        P = F * track.P * F.transpose() + self.Q()  # 오차 공분산 예측
        track.set_x(x)
        track.set_P(P)
 
         ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############

        '''상태 및 공분산 업데이트'''
        H = meas.sensor.get_H(track.x)  # 관측 행렬
        gamma = self.gamma(track, meas)  # 잔차 계산
        S = self.S(track, meas, H)  # 잔차의 공분산 계산
        K = track.P * H.transpose() * np.linalg.inv(S)  # 칼만 이득 계산
        x = track.x + K * gamma  # 상태 업데이트
        I = np.identity(params.dim_state)  # 단위 행렬
        P = (I - K * H) * track.P  # 공분산 업데이트
        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)
            
        ############
        # END student code
        ############ 
        #track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
        #opt1
        #잔차 계산: 측정값과 예측값의 차이
        #z = meas.z
        #hx = meas.sensor.get_hx(track.x)  # 예측 측정값
        #gamma = z - hx
        #return gamma
        ###############
        #Opt2
        '''잔차 (측정과 예측의 차이) 계산'''
        gamma = meas.z - meas.sensor.get_hx(track.x)
        return gamma

        

        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
        #Opt1
         # 잔차 공분산 S 계산
        #return H * track.P * H.T + meas.R
               
        ############
        #Opt2
        '''잔차의 공분산 계산'''
        S = H * track.P * H.transpose() + meas.R
        return S


        # END student code
        ############ 