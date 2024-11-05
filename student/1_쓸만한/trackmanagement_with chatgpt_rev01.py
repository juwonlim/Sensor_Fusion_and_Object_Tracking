# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#


#최종프로젝트
#중간과제로부터 업데이트된 항목
#트랙의 히스토리(self.history) 추가: 트랙에 히스토리를 저장하는 기능을 추가. 이는 각 트랙이 업데이트될 때마다 히스토리에 측정값을 추가하도록 구현.
#트랙 상태 예측 로직: predict_next_state 메서드에서 Mabhi16의 코드에서 추가된 부분을 반영할 준비가 되어 있음


# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params 



class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id):
        print('creating track no.', id)
        self.id = id  # 트랙의 고유 ID
        self.x = meas.x  # 상태 벡터 초기화
        self.P = meas.P  # 공분산 행렬 초기화
        self.state = 'initialized'  # 트랙의 초기 상태
        self.score = 1.0  # 초기 트랙 점수
        self.timestamp = meas.timestamp  # 마지막 업데이트 시간
        self.missed_count = 0  # 누락된 업데이트 횟수 초기화
        self.history = []  # 트랙의 히스토리 저장

    def set_x(self, x):
        '''상태 벡터 설정'''
        self.x = x

    def set_P(self, P):
        '''공분산 행렬 설정'''
        self.P = P

    def update_attributes(self, meas):
        '''측정값으로 트랙 속성 업데이트'''
        self.x = meas.x  # 새로운 상태 벡터로 업데이트
        self.P = meas.P  # 새로운 공분산 행렬로 업데이트
        self.timestamp = meas.timestamp  # 업데이트된 시간 기록
        self.score += 1.0  # 트랙 점수 증가
        self.missed_count = 0  # 누락된 횟수 초기화
        self.state = 'confirmed' if self.score >= params.confirmation_threshold else 'tentative'  # 상태 업데이트
        self.history.append(meas)  # 트랙 히스토리에 측정값 추가

    #추가한 함수 
    def increase_missed_count(self):
        '''누락된 업데이트 횟수 증가'''
        self.missed_count += 1
        print(f'Track {self.id}: missed count increased to {self.missed_count}')
        if self.missed_count > params.deletion_threshold:
            self.state = 'deleted'  # 상태를 삭제로 변경해 삭제 조건 확인

    #추가한 함수
    def should_be_deleted(self):
        '''트랙 삭제 여부 판단'''
        return self.state == 'deleted'  # 상태가 'deleted'인지 확인

    #추가한 함수
    def predict_next_state(self):
        '''트랙의 다음 상태 예측'''
        print(f'Track {self.id}: predicting next state')
        # 상태 예측에 필요한 로직 추가 가능



class TrackManagement:
    def __init__(self):
        self.tracks = []  # 활성화된 트랙 리스트
        self.next_id = 0  # 새 트랙에 할당할 ID

    def manage_tracks(self, unassigned_tracks, measurements):
        '''할당되지 않은 트랙 업데이트 및 새로운 측정값으로 트랙 추가'''
        for track in unassigned_tracks:
            track.increase_missed_count()  # 업데이트되지 않은 트랙의 누락 카운트 증가
            if track.should_be_deleted():
                self.delete_track(track)  # 누락 횟수가 초과된 경우 삭제
            else:
                track.predict_next_state()  # 삭제되지 않는 경우 상태 예측

        for meas in measurements:
            if meas.is_unassigned():
                self.add_track_to_list(meas)  # 새 측정값으로 새 트랙 추가

    def add_track_to_list(self, meas):
        '''새 측정값을 바탕으로 새 트랙 추가'''
        new_track = Track(meas, self.next_id)
        self.tracks.append(new_track)
        self.next_id += 1
        print(f'New track initialized: ID {new_track.id}')

    def delete_track(self, track):
        '''트랙 삭제'''
        self.tracks.remove(track)
        print(f'Track deleted: ID {track.id}')

    def handle_updated_track(self, track, meas):
        '''업데이트된 트랙 처리'''
        track.update_attributes(meas)  # 트랙 속성 업데이트
