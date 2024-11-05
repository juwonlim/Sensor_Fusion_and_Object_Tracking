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
        '''시스템 행렬 F를 반환'''
        # 시스템 행렬 정의
        # Mabhi16의 코드에서 사용된 추가 논리를 참고해 구현
        dt = params.dt
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        return F

    def Q(self):
        '''프로세스 노이즈 공분산 행렬 Q를 반환'''
        # 노이즈 공분산 행렬 정의
        q = params.q
        dt = params.dt
        Q = np.array([[dt**4/4*q, 0, dt**3/2*q, 0],
                      [0, dt**4/4*q, 0, dt**3/2*q],
                      [dt**3/2*q, 0, dt**2*q, 0],
                      [0, dt**3/2*q, 0, dt**2*q]])
        return Q

    def predict(self, x, P):
        '''상태 벡터 x와 공분산 행렬 P를 예측'''
        F = self.F()
        x_pred = np.dot(F, x)  # 예측된 상태 벡터
        P_pred = np.dot(F, np.dot(P, F.T)) + self.Q()  # 예측된 공분산 행렬
        return x_pred, P_pred

    def update(self, x, P, z, H, R):
        '''측정값 z를 사용해 상태 벡터 x와 공분산 행렬 P를 업데이트'''
        y = z - np.dot(H, x)  # 측정 잔차
        S = np.dot(H, np.dot(P, H.T)) + R  # 잔차 공분산
        K = np.dot(P, np.dot(H.T, np.linalg.inv(S)))  # 칼만 이득
        x_upd = x + np.dot(K, y)  # 업데이트된 상태 벡터
        P_upd = P - np.dot(K, np.dot(H, P))  # 업데이트된 공분산 행렬
        return x_upd, P_upd

    def gamma(self, x, z, H):
        '''측정 잔차 계산'''
        return z - np.dot(H, x)

    def S(self, P, H, R):
        '''잔차 공분산 행렬 S 계산'''
        return np.dot(H, np.dot(P, H.T)) + R
