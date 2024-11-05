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


#최종프로젝트 제출요구사항:
#할당되지 않은 측정의 트랙 초기화 구현:
#add_track_to_list 함수에서 구현, 이 함수는 할당되지 않은 측정을 받아 새로운 Track 객체를 초기화하고 리스트에 추가

#트랙 점수 정의 및 구현:
#이 코드에는 트랙 점수가 명시적으로 구현되지 않았지만, 점수는 추적 객체의 신뢰도를 나타낼 수 있는 속성으로 추가할 수 있음. 
#추가적인 코드 변경이 필요하다면 Track 클래스에 score 속성을 추가하고 업데이트 메서드에서 관리할 수 있음.

#트랙 상태(“잠정적”, “확인됨”) 정의 및 구현:
#현재 상태는 Track 클래스에 포함되어 있지 않지만, 상태 관리를 위해 status 속성을 추가하여 “잠정적” 또는 “확인됨” 상태로 관리할 수 있음.

#이전 트랙 삭제:
#manage_tracks 메서드에서 track.should_be_deleted() 조건이 참일 경우 delete_track 함수를 호출하여 이전 트랙이 삭제되도록 구현되어 있음

#


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
    """상태 및 공분산 관리를 위한 트랙 클래스."""
    def __init__(self, measurement, track_id):
        # 트랙 ID가 주어질 때 초기화 메시지 출력
        print(f"트랙 ID {track_id} 생성 중")
        self.id = track_id  # 트랙의 고유 식별자 설정
        self.measurements = [measurement]  # 첫 번째 측정값을 트랙에 저장
        self.missed_count = 0  # 트랙 업데이트가 누락된 횟수를 기록
        
        # 측정값에 상태 벡터와 공분산 행렬이 있는 경우 초기화, 없으면 None으로 설정
        self.x = measurement.x if hasattr(measurement, 'x') else None  # 초기 상태 벡터
        self.P = measurement.P if hasattr(measurement, 'P') else None  # 초기 공분산 행렬
        
        # 측정값에 타임스탬프가 있으면 마지막 업데이트 시간으로 설정, 없으면 None
        self.t = measurement.timestamp if hasattr(measurement, 'timestamp') else None  # 마지막 업데이트 시간

    def set_x(self, x):
        """상태 벡터 설정.
        주어진 상태 벡터 x를 트랙에 설정.
        """
        self.x = x

    def set_P(self, P):
        """공분산 행렬 설정.
        주어진 공분산 행렬 P를 트랙에 설정.
        """
        self.P = P

    def set_t(self, t):
        """마지막 업데이트 시간 설정.
        주어진 시간 t를 트랙의 마지막 업데이트 시간으로 설정.
        """
        self.t = t

    def update_attributes(self, measurement):
        """새로운 측정값으로 트랙 속성 업데이트.
        새로운 측정값을 추가하고, 누락된 업데이트 횟수를 초기화.
        """
        self.measurements.append(measurement)  # 새 측정값을 리스트에 추가
        self.missed_count = 0  # 업데이트 시 누락된 횟수를 0으로 초기화

    def increase_missed_count(self):
        """누락된 업데이트 횟수 증가.
        트랙이 업데이트되지 않은 경우, 누락된 횟수를 1 증가시킴.
        """
        self.missed_count += 1
        print(f"트랙 ID {self.id}: 누락 횟수 {self.missed_count}로 증가")

    def should_be_deleted(self):
        """누락된 업데이트 횟수를 기준으로 트랙 삭제 여부 결정.
        일정 횟수 이상 업데이트가 누락되었을 때 트랙을 삭제할지 결정함.
        """
        return self.missed_count > 5  # 삭제 임계값, 필요에 따라 조정 가능

    def predict_next_state(self):
        """트랙의 다음 상태 예측.
        트랙의 현재 상태를 기반으로 다음 상태를 예측.
        """
        print(f"트랙 ID {self.id}: 다음 상태 예측 중")


class TrackManagement:
    def __init__(self):
        # 활성화된 트랙 리스트 초기화
        self.tracks = []  # 현재 관리 중인 트랙 리스트
        self.next_id = 0  # 다음에 생성될 트랙에 할당될 ID

    def manage_tracks(self, unassigned_tracks, measurements):
        """할당되지 않은 트랙 업데이트 및 새로운 측정값 추가로 트랙 관리.
        트랙이 업데이트되지 않은 경우 누락 횟수를 증가시키고 필요 시 삭제하며,
        새로운 측정값을 통해 새 트랙을 생성.
        """
        for track in unassigned_tracks:
            track.increase_missed_count()  # 할당되지 않은 트랙의 누락 횟수 증가
            if track.should_be_deleted():  # 누락 횟수가 임계값을 초과하면 트랙 삭제
                self.delete_track(track)
            else:
                track.predict_next_state()  # 삭제되지 않는 경우 다음 상태 예측

        for measurement in measurements:
            if measurement.is_unassigned():  # 할당되지 않은 측정값인지 확인
                self.add_track_to_list(measurement)  # 새 트랙 생성

    def add_track_to_list(self, measurement):
        """새 측정값으로 새로운 트랙 초기화 및 리스트에 추가.
        새로운 트랙을 생성하여 트랙 리스트에 추가.
        """
        new_track = Track(measurement, self.next_id)  # 새 트랙 객체 생성
        self.tracks.append(new_track)  # 트랙 리스트에 추가
        self.next_id += 1  # 다음 트랙 ID 증가
        print(f"새 트랙 초기화: ID {new_track.id}")

    def delete_track(self, track):
        """트랙 리스트에서 지정된 트랙 삭제.
        특정 트랙을 리스트에서 제거하고 삭제 메시지를 출력.
        """
        self.tracks.remove(track)
        print(f"트랙 삭제: ID {track.id}")

    def handle_updated_track(self, track, measurement):
        """업데이트된 트랙 처리하여 속성 업데이트.
        기존 트랙의 속성을 새로운 측정값으로 갱신.
        """
        track.update_attributes(measurement)
