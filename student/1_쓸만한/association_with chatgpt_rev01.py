# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params 

class Association:
    '''최인접 항목 데이터 연결과 마할라노비스 거리 기반 게이팅을 위한 데이터 연관 클래스'''

    def __init__(self):
        # 연결 행렬: 트랙과 측정값 간의 연관성을 나타내며 마할라노비스 거리 값을 포함
        self.association_matrix = np.matrix([])  # 기존 구조 유지, 빈 행렬로 초기화
        # 할당되지 않은 트랙 인덱스를 저장하는 리스트
        self.unassigned_tracks = []
        # 할당되지 않은 측정값 인덱스를 저장하는 리스트
        self.unassigned_meas = []

    def associate(self, track_list, meas_list, KF):
        '''
        트랙 리스트와 측정값 리스트를 입력으로 받아 연결 행렬 계산
        각 트랙과 측정값 간 마할라노비스 거리 계산 후 연결 행렬 업데이트
        할당되지 않은 트랙과 측정값 인덱스 초기화
        '''
        num_tracks = len(track_list)
        num_meas = len(meas_list)
        self.association_matrix = np.inf * np.ones((num_tracks, num_meas))  # 연결 행렬을 무한대로 초기화
        
        for i, track in enumerate(track_list):
            for j, meas in enumerate(meas_list):
                dist = self.MHD(track, meas, KF)  # 마할라노비스 거리 계산
                if self.gating(dist, meas.sensor):  # 게이팅 조건 만족 여부 확인
                    self.association_matrix[i, j] = dist  # 연결 행렬에 거리 값 저장

        self.unassigned_tracks = list(range(num_tracks))  # 할당되지 않은 트랙 인덱스 저장
        self.unassigned_meas = list(range(num_meas))  # 할당되지 않은 측정값 인덱스 저장

    def get_closest_track_and_meas(self):
        '''
        연결 행렬에서 최인접 트랙과 측정값의 인덱스 반환
        반환 후 해당 트랙과 측정값 인덱스를 리스트에서 제거
        '''
        if np.min(self.association_matrix) == np.inf:
            return np.nan, np.nan  # 유효한 연결이 없으면 NaN 반환

        track_idx, meas_idx = np.unravel_index(np.argmin(self.association_matrix), self.association_matrix.shape)
        self.association_matrix[track_idx, :] = np.inf  # 선택된 트랙의 행을 무효화해 다른 연결 방지
        self.association_matrix[:, meas_idx] = np.inf  # 선택된 측정값의 열을 무효화해 다른 연결 방지

        self.unassigned_tracks.remove(track_idx)  # 할당된 트랙 인덱스 제거
        self.unassigned_meas.remove(meas_idx)  # 할당된 측정값 인덱스 제거

        return track_idx, meas_idx

    def gating(self, MHD, sensor):
        '''
        마할라노비스 거리의 게이팅 조건 확인
        카이 제곱 분포를 사용해 거리 제한 값과 비교해 게이팅 조건 판별
        '''
        limit = chi2.ppf(params.gating_threshold, df=sensor.dim_meas)  # 카이 제곱 분포의 임계값 계산
        return MHD < limit  # 거리 제한과 비교해 결과 반환

    def MHD(self, track, meas, KF):
        '''
        트랙과 측정값 간 마할라노비스 거리 계산
        측정값과 트랙 상태 벡터 간 차이(gamma)와 공분산 행렬(S)을 이용해 거리 계산
        '''
        H = meas.sensor.get_H(track.x)  # 관측 행렬 계산
        S = KF.S(track, meas, H)  # 잔차 공분산 계산
        gamma = KF.gamma(track, meas)  # 잔차 계산
        return gamma.T @ np.linalg.inv(S) @ gamma  # 마할라노비스 거리 반환
