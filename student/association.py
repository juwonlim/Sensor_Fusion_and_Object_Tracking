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
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self): #역할: 객체를 초기화하고 association_matrix, unassigned_tracks, unassigned_meas 등의 변수를 준비
        self.association_matrix = np.matrix([])  #association_matrix: 트랙과 측정값 간의 연관성을 나타내는 매트릭스.이 매트릭스는 Mahalanobis 거리 값을 포함
        self.unassigned_tracks = [] #unassigned_tracks: 아직 연관되지 않은 트랙의 인덱스를 저장하는 리스트
        self.unassigned_meas = [] #unassigned_meas: 아직 연관되지 않은 측정값의 인덱스를 저장하는 리스트
        
    def associate(self, track_list, meas_list, KF): #주어진 트랙 리스트와 측정값 리스트 사이의 Mahalanobis 거리를 계산하여 association_matrix를 업데이트
                                                    #track_list: 현재 추적 중인 트랙 리스트
                                                    #meas_list: 새로운 측정값 리스트
                                                    #KF: 칼만 필터 객체.
        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        ############
        # Step 3: association with weight application:
        # - calculate Mahalanobis distance between tracks and measurements
        # - apply a weight factor based on the distance (closer measurements get more weight)
        #Mahalanobis거리에 가중치를 적용하는 부분 추가 , 가까운 물체에 더 큰 가중치 적용
        #트랙과 측정값간의 매칭을 개선할 수 있도록 설계
        ############
        
        N = len(track_list) 
        M = len(meas_list) 
        self.association_matrix = np.ones((N, M)) * np.inf #트랙과 측정값 간의 Mahalanobis 거리 값을 저장.
                                                           #Mahalanobis 거리가 특정 임계값 이내에 있는 경우, 트랙과 측정값을 연관시키고, 그렇지 않은 경우 연관되지 않음.
        # Define the maximum distance threshold for gating
        max_dist_threshold = 100  # adjust based on application needs
        
        for i in range(N):
            track: Track = track_list[i]
            for j in range(M):
                meas: Measurement = meas_list[j]
                dist = self.MHD(track, meas, KF)  # Mahalanobis distance calculation, 가장 가까운 이옷(nearest neighbor)을 찾기위한 거리계산
                
                if self.gating(dist, meas.sensor): #거리를 기반으로 문턱값(gating)을 적용하여 트랙과 측정값을 연결할지 여부를 결정
                    # Apply weight based on distance
                    # Inverse distance as weight (closer measurements have higher weights)
                    weight = np.exp(-dist / max_dist_threshold)  #이 코드를 이용하여 거리 기반 가중치 계산, 가까운 측정값일 수록 더 높은 가중치가 적용됨, 
                                                                 #max_dist_threshold는 가중치 제어하는 매개변수, 더 작은 값을 설정하면 가까운 거리에 더 큰 가중치 적용
                    weighted_dist = dist * weight  # Apply weight to the distance , 계산된 mahalanobis distance에 가중치를 적용해 weighted_dist를 계산,이를 최종적으로 association_matrix에 저장
                    
                    self.association_matrix[i, j] = weighted_dist #weihted_dist를 association_matrix에 저장
                                                    #가까운 측정값에 더 높은 가중치를 주기 위해 가중치(weight)를 사용한 거리 계산이 추가됨.
        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))






        ''' 
        # the following only works for at most one track and one measurement
        self.association_matrix = np.matrix([]) # reset matrix
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []
        
        if len(meas_list) > 0:
            self.unassigned_meas = [0]
        if len(track_list) > 0:
            self.unassigned_tracks = [0]
        if len(meas_list) > 0 and len(track_list) > 0: 
            self.association_matrix = np.matrix([[0]])
        
        '''
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self): 
        #역할: association_matrix에서 가장 작은 값을 찾아 그 트랙과 측정값을 반환
        #내부로직: 가장 작은 거리 값을 찾고 해당 트랙과 측정값을 unassigned_tracks와 unassigned_meas 리스트에서 제거 ,선택된 트랙과 측정값의 인덱스를 반환.
                 
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############
         # Find the closest track and measurement based on the association matrix
        min_value = np.min(self.association_matrix)
        if np.isinf(min_value):
            return np.nan, np.nan
        
        # Find the indexes of the minimum value in the association matrix
        track_idx, meas_idx = np.unravel_index(np.argmin(self.association_matrix, axis=None), self.association_matrix.shape)
        
        # Remove the associated track and measurement from the unassigned lists
        self.unassigned_tracks.remove(track_idx)
        self.unassigned_meas.remove(meas_idx)
        
        # Set the corresponding value in the association matrix to infinity to mark it as used
        self.association_matrix[track_idx, meas_idx] = np.inf
        
        return track_idx, meas_idx



        ''' 
        # the following only works for at most one track and one measurement
        update_track = 0
        update_meas = 0
        
        # remove from list
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
        self.association_matrix = np.matrix([])
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     
        '''
    def gating(self, MHD, sensor): #역할: Mahalanobis 거리를 기준으로 주어진 측정값이 트랙과 연관될 수 있는지 확인
                                   #MHD: Mahalanobis 거리 값.
                                   #sensor: 측정값을 제공하는 센서 객체.
        #내부로직: Mahalanobis 거리가 센서의 임계값을 넘지 않는 경우 True를 반환하여 트랙과 측정값을 연관
        
        ############
        # TODO Step 3: return True if measurement lies inside gate, otherwise False
        ############
        """Gating based on Mahalanobis distance and sensor type"""
        # Define gating thresholds based on the sensor type
        threshold = params.gating_threshold[sensor.name]  # assume there's a threshold in params for each sensor
        return dist < threshold
           
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF): #역할: 트랙과 측정값 간의 Mahalanobis 거리를 계산
        #track: 트랙 객체, meas: 측정값 객체, KF: 칼만 필터 객체.
        ############
        # TODO Step 3: calculate and return Mahalanobis distance
        ############
        """Mahalanobis distance calculation"""
        #내부로직:
        #H:측정 매트릭스
        #S:잔차의 공분산 행렬
        #gamma: 측정값과 예측된 값 간의 잔차


        H = meas.sensor.get_H(track.x)  # measurement matrix
        S = H * KF.P * H.T + meas.R  # covariance of residual
        gamma = meas.z - meas.sensor.get_hx(track.x)  # residual
        
        dist = gamma.T * np.linalg.inv(S) * gamma  # Mahalanobis distance
                                                   #Mahalanobis 거리를 계산하고 반환.
        
        return dist.item()
        
        ############
        # END student code
        ############ 
    
    #associate_and_update 함수, 역할: 트랙과 측정값을 연관시키고, 연관된 트랙에 대해 칼만 필터를 사용하여 업데이트
    def associate_and_update(self, manager, meas_list, KF):
        #manager: 트랙 관리 객체, meas_list: 측정값 리스트, KF: 칼만 필터 객체
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        #내부로직 : 연관되지 않은 트랙과 측정값이 있는 동안, 가장 가까운 트랙과 측정값을 찾아서 업데이트를 수행.
        #내부로직2 :  업데이트된 트랙의 상태를 갱신하고, 점수를 기록.        
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)