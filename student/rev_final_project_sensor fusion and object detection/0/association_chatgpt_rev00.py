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
#할당되지 않은 측정의 트랙 초기화: add_track() 메서드에서 할당되지 않은 측정값을 기반으로 새로운 트랙이 생성되도록 구현
#트랙 점수 정의 및 구현: Track 클래스에는 점수 대신 missed_count로 누락된 업데이트를 관리하며, 일정 임계값 이상이면 삭제
#트랙 상태 정의 및 구현: increase_missed_count()와 should_be_deleted()를 통해 상태를 관리하며, 해당 로직으로 '잠정적' 상태에서 '확인됨' 또는 '삭제' 상태로의 전환이 가능함
#업데이트되지 않은 트랙 삭제: update_tracks() 메서드에서 트랙이 업데이트되지 않으면 missed_count를 증가시키고 should_be_deleted()로 삭제 조건을 확인


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
    def __init__(self, measurement, track_id):
        """
        트랙 클래스 초기화.
        """
        self.id = track_id  # 트랙 ID
        self.measurements = [measurement]  # 초기 측정값 저장
        self.missed_count = 0  # 누락된 업데이트 카운트
        self.x = None  # 상태 벡터
        self.P = None  # 상태 공분산 행렬
        self.t = None  # 마지막 업데이트 시간

    def set_x(self, x):
        """
        상태 벡터를 설정합니다.
        """
        self.x = x

    def set_P(self, P):
        """
        상태 공분산 행렬을 설정합니다.
        """
        self.P = P

    def set_t(self, t):
        """
        마지막 업데이트 시간을 설정합니다.
        """
        self.t = t

    def update_attributes(self, measurement):
        """
        트랙의 속성을 새로운 측정값으로 업데이트합니다.
        """
        self.measurements.append(measurement)
        self.missed_count = 0  # 새로운 측정값이 들어오면 누락 카운트 리셋

    def increase_missed_count(self):
        """
        누락된 업데이트 카운트를 증가시킵니다.
        """
        self.missed_count += 1
        print(f"트랙 ID {self.id}: 누락 카운트 증가, 현재 {self.missed_count}")

    def should_be_deleted(self):
        """
        트랙이 삭제될 조건을 확인합니다.
        """
        return self.missed_count > 5  # 임계값은 필요에 따라 조정 가능

    def predict_next_state(self):
        """
        다음 상태를 예측합니다.
        """
        print(f"트랙 ID {self.id}: 다음 상태 예측")

class TrackManagement:
    def __init__(self):
        # 트랙을 관리하기 위한 초기화 코드
        self.tracks = []  # 현재 활성화된 트랙 리스트
        self.next_id = 0  # 새 트랙에 할당될 ID 값 초기화

    def manage_tracks(self, unassigned_tracks, measurements):
        """
        할당되지 않은 트랙을 업데이트하거나 삭제하고, 새 측정값을 추가합니다.
        """
        for track in unassigned_tracks:
            track.increase_missed_count()  # 업데이트되지 않은 트랙의 누락 카운트 증가
            if track.should_be_deleted():  # 누락 카운트가 특정 임계값을 초과하면 삭제
                self.delete_track(track)
            else:
                track.predict_next_state()  # 예측 단계 수행

        for measurement in measurements:
            if measurement.is_unassigned():
                self.addTrackTolist(measurement)  # 새로운 측정값으로 새 트랙 추가

    def addTrackTolist(self, measurement):
        """
        새로운 측정을 바탕으로 새로운 트랙을 초기화합니다.
        """
        new_track = Track(measurement, self.next_id)
        self.tracks.append(new_track)
        self.next_id += 1
        print(f"새 트랙 초기화: ID {new_track.id} 생성됨")

    def init_track(self, measurement):
        """
        트랙 초기화 로직을 처리합니다.
        """
        self.addTrackTolist(measurement)

    def delete_track(self, track):
        """
        지정된 트랙을 삭제합니다.
        """
        self.tracks.remove(track)
        print(f"트랙 삭제: ID {track.id} 제거됨")

    def handle_updated_track(self, track, measurement):
        """
        업데이트된 트랙을 처리합니다.
        """
        track.update_attributes(measurement)  # 속성 업데이트
