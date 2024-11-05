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
F(): 시스템 행렬을 반환하는 함수. 시스템의 시간 변화를 반영한 상태 변화를 나타냄
Q(): 프로세스 노이즈 공분산 행렬을 반환. 상태의 예측에서 불확실성을 반영
predict(): 상태와 공분산을 다음 시점으로 예측하는 과정.
update(): 측정된 데이터를 사용해 상태와 공분산을 업데이트
gamma(): 측정 잔차(측정값과 예측값의 차이)를 계산.
S(): 잔차의 공분산 행렬을 계산하여 업데이트 과정에 사용
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
        pass

    def F(self):
        ############
        # TODO Step 1: implement and return system matrix F
        ############
        # 시스템 행렬 F를 구현
        # 시스템의 상태 변화를 나타내는 행렬을 반환
        dt = params.dt  # time delta
        F = np.matrix([[1, 0, dt, 0, 0, 0],
                       [0, 1, 0, dt, 0, 0],
                       [0, 0, 1, 0, dt, 0],
                       [0, 0, 0, 1, 0, dt],
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
        return F


        #return 0
        ############
        # END student code
        ############ 

    def Q(self):
        ############
        # TODO Step 1: implement and return process noise covariance Q
        ############
        # 프로세스 노이즈 공분산 행렬 Q를 구현
        # 시스템의 불확실성을 고려하는 행렬을 반환
        ''' 
        #mid term제출코드

        q = params.q
        dt = params.dt
        Q = q * np.matrix([[dt**3 / 3, 0, dt**2 / 2, 0, 0, 0],
                           [0, dt**3 / 3, 0, dt**2 / 2, 0, 0],
                           [dt**2 / 2, 0, dt, 0, 0, 0],
                           [0, dt**2 / 2, 0, dt, 0, 0],
                           [0, 0, 0, 0, dt, 0],
                           [0, 0, 0, 0, 0, dt]])
        return Q
        '''
        #mabhi의 코드응용
        q = params.q
        dt = params.dt
        q1 = ((dt**3)/3) * q 
        q2 = ((dt**2)/2) * q 
        q3 = dt * q 
        Q = np.matrix([[q1, 0, 0, q2, 0, 0],
                          [0, q1, 0, 0, q2, 0],
                          [0, 0, q1, 0, 0, q2],
                          [q2, 0, 0, q3, 0, 0],
                          [0, q2, 0, 0, q3, 0],
                          [0, 0, q2, 0, 0, q3]])
    
        return Q



        #return 0
        ############
        # END student code
        ############ 

    def predict(self, track):
        ############
        # TODO Step 1: predict state x and estimation error covariance P to next timestep, save x and P in track
        ############
        ''' 
        #mid term 제출
        # 상태 x와 공분산 P를 다음 시점으로 예측하여 track에 저장 
        #미래의 상태를 예측한 후 track 객체에 저장된 x(상태벡터)와 p(오차 공분산 행렬)을 (직접)갱신, 그래서 no return 값
        F = self.F()
        track.x = F * track.x  # 상태 예측
        track.P = F * track.P * F.T + self.Q()  # 공분산 예측
        '''
        #final project, mabhi코드 응용
        F = self.F()
        x = track.x
        P = track.P
        x = F*track.x # state prediction
        P = F*track.P*F.transpose() + self.Q() # covariance prediction
        track.set_x(x)
        track.set_P(P)



       #pass
        ############
        # END student code
        ############ 

    def update(self, track, meas):
        ############
        # TODO Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        # 측정값을 사용하여 상태 x와 공분산 P를  (직접) 업데이트 그래서 no return값
        H = meas.sensor.get_H(track.x)  # 측정 행렬
        gamma = self.gamma(track, meas)  # 측정 잔차
        S = self.S(track, meas, H)  # 잔차 공분산
        K = track.P * H.T * np.linalg.inv(S)  # 칼만 이득
        track.x = track.x + K * gamma  # 상태 업데이트
        I = np.identity(params.dim_state)
        track.P = (I - K * H) * track.P  # 공분산 업데이트
        track.update_attributes(meas)  # 추가적인 속성 업데이트

        
        ############
        # END student code
        ############ 
        #track.update_attributes(meas)
    
    def gamma(self, track, meas):
        ############
        # TODO Step 1: calculate and return residual gamma
        ############
         # 잔차 계산: 측정값과 예측값의 차이
        z = meas.z
        hx = meas.sensor.get_hx(track.x)  # 예측 측정값
        gamma = z - hx
        return gamma

        #return 0
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # TODO Step 1: calculate and return covariance of residual S
        ############
         # 잔차 공분산 S 계산
        return H * track.P * H.T + meas.R
        
        #return 0
        ############
        # END student code
        ############ 