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
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        ############
        # TODO Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on 
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############
        '''mid term 과제 코드 

         ############
        # - 고정 초기값을 사용하여 x와 P를 초기화
        # - 트랙의 상태와 점수를 적절히 초기화
        ############


        self.x = np.matrix([[49.53980697],
                        [ 3.41006279],
                        [ 0.91790581],
                        [ 0.        ],
                        [ 0.        ],
                        [ 0.        ]])
        self.P = np.matrix([[9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
                        [0.0e+00, 9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
                        [0.0e+00, 0.0e+00, 6.4e-03, 0.0e+00, 0.0e+00, 0.0e+00],
                        [0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00, 0.0e+00],
                        [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00],
                        [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+01]])

        # 트랙 상태와 점수 초기화
        self.state = 'initialized'  # 초기화 상태
        self.score = 1  # 초기 점수 설정

        #self.state = 'confirmed'
        #self.score = 0
        
        ############
        # END student code
        ############ 
               
        # other track attributes
         # 트랙의 다른 속성들
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t
        '''
        #mabhi의 코드 응용
        pos_sens = np.ones((4, 1))
        pos_sens[0:3] = meas.z[0:3] 
        pos_veh = meas.sensor.sens_to_veh*pos_sens
        self.x = np.zeros((6,1))
        self.x[0:3] = pos_veh[0:3]


        P_pos = M_rot * meas.R * np.transpose(M_rot)
        P_vel = np.matrix([[params.sigma_p44**2, 0, 0],
                           [0, params.sigma_p55**2, 0],
                           [0, 0, params.sigma_p66**2]])
        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel
        self.state = 'initialized'
        self.score = 1/params.window

         # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        self.t = meas.t



        

    def set_x(self, x):
        self.x = x
        
    def set_P(self, P):
        self.P = P  
        
    def set_t(self, t):
        self.t = t  
        
    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        
        
###################        

class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############
         ############
        # - 할당되지 않은 트랙의 점수를 줄임
        # - 트랙의 점수가 너무 낮거나 공분산 행렬 P가 너무 큰 경우 트랙을 삭제
        ############
        
        # decrease score for unassigned tracks
        # 할당되지 않은 트랙의 점수를 줄임
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility
            # 측정값의 시야에 있는지 확인   
            if meas_list and meas_list[0].sensor.in_fov(track.x):
               track.score -= 1 / params.window #멘토의 코드 (멘토는 1을 빼고 줬으나 '/'앞에 숫자와야함.chatgpt조언)
               #track.score -= 1  # 점수를 줄임 -->기존코드 

            '''
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    # your code goes here
                    pass 
            '''
        # delete old tracks
        ''' 
        #문제가 되었던 코드
        # 오래된 트랙 삭제
        for track in self.track_list:
            if track.score < params.delete_threshold or np.max(track.P.diagonal()) > params.max_P:
                self.delete_track(track)
        '''

        #멘토님의 코드
        #manage_tracks 함수에서 현재 track의 score가 일정값 이하면 state와 상관없이 그냥 삭제가 되고 있습니다.
        #아래처럼 confirm이 된 track일 경우에만 삭제를 해주셔야 합니다. 아니면 track이 score가 쌓이기도 전에 바로 삭제가 되버립니다.
       
        '''
        멘토가 제공한 "오래된 트랙 삭제" 코드는 추적 관리 시스템에서 오래된(낮은 점수를 가진) 트랙을 삭제하기 위해 설계된 것입니다. 
        이 코드는 특정 조건에 따라 트랙을 확인하고 필요하면 삭제합니다. 여기서 의미하는 바는 다음과 같습니다:
        deleted_tracks 리스트 생성:

        deleted_tracks라는 빈 리스트를 생성하여 삭제해야 할 트랙을 저장합니다.
        할당되지 않은 트랙 순회:

        for i in unassigned_tracks: 루프는 unassigned_tracks 리스트를 순회합니다. 
        unassigned_tracks는 현재 매칭되지 않은 트랙의 인덱스를 포함하는 리스트일 수 있습니다.
        트랙 조건 확인:

        각 트랙을 확인하여 track.state가 "confirmed"인 경우와 track.score가 params.delete_threshold보다 
        작은 경우에 해당 트랙을 deleted_tracks 리스트에 추가합니다.
        여기서 track.state == "confirmed"는 해당 트랙이 확인된 상태(신뢰할 수 있는 상태)임을 의미합니다.
        track.score < params.delete_threshold는 해당 트랙이 특정 임계값 미만의 점수를 가진 경우를 나타냅니다. 
        즉, 트랙이 충분히 신뢰할 수 없거나 오래되었을 수 있습니다.
        트랙 삭제:

        for track in deleted_tracks: 루프는 삭제해야 할 트랙을 실제로 삭제하는 작업을 수행합니다.
        self.delete_track(track)은 해당 트랙을 관리하는 목록이나 데이터 구조에서 트랙을 제거합니다.
        이 코드의 목적은 오래되거나 신뢰할 수 없는 트랙을 제거하여 추적 관리 시스템을 정리하고 더 신뢰할 수 있는 트랙만 유지하도록 하는 것입니다. 
        이 방식은 객체 추적 알고리즘에서 자주 사용되는 개념으로, 추적이 오래되었거나 더 이상 유효하지 않을 때 시스템 리소스를 확보하고 정확성을 높이기 위한 것입니다.
        '''
        
       # 오래된 트랙 삭제
        delete_tracks = []
        #멘토님 두번째 코드
        for i in range(len(self.track_list)):
            track = self.track_list[i]
            if track.state == 'confirmed' and track.score < params.delete_threshold:
                delete_tracks.append(track)
                continue
            if track.score < 0.15:
                delete_tracks.append(track)
                continue
            if track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P:
                delete_tracks.append(track)


        ''' 
        #멘토님 첫번째 코드 --> 폐기됨
        for i in unassigned_tracks:
            track = self.track_list[i]
            if track.state == "confirmed" and track.score < params.delete_threshold:
                deleted_tracks.append(track)
                
        for track in deleted_tracks:
            self.delete_track(track)
            '''
        

                    
        

        ############
        # END student code
        ############ 
            
        # initialize new track with unassigned measurement
         # 새로운 트랙을 초기화
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements ,# Lidar 측정값으로만 초기화
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        #새로운 측정을 기반으로 트랙을 초기화
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

        # chatgpt - 디버깅: 새로 생성된 트랙의 초기 값 출력
        print(f"Initialized Track ID: {track.id}")
        print(f"Initial x: {track.x}")
        print(f"Initial P: {track.P}")
        print(f"Initial State: {track.state}")
        print(f"Initial Score: {track.score}")

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)
        
    def handle_updated_track(self, track):      
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############
        ############
        # - 트랙 점수를 증가시킴
        # - 트랙 상태를 'tentative' 또는 'confirmed'로 설정
        ############
        ''' 
        #https://github.com/jckuri/Sensor-Fusion-and-Object-Tracking-2/blob/main/student/trackmanagement.py#L118
        #jckuri의 코드
        track.append_assignment_and_compute_score(1, cnt_frame)
        if track.state == 'initialized' and track.score >= 0.8:
            track.state = 'tentative'
        if track.state == 'tentative' and track.score >= 1.5:
            track.state = 'confirmed'
        '''
       
        
        #여기서부터 멘토가 준 코드
        track.score += 1. / params.window  # 트랙 점수 증가

        # CHATGPT추천-디버깅: 트랙의 상태와 공분산 값(P[0,0]과 P[1,1]) 출력
        # 트랙이 지나치게 빠르게 삭제되지 않도록 하기 위해 𝑃[0,0],,𝑃[1,1] 값이 params.max_P 조건을 만족하는지 확인
        # 만약 P 값이 너무 크다면, 이는 트랙이 불안정하다는 의미로 간주
        # P[0,0], P[1,1] 값이 지나치게 크면: params.max_P 값을 조정하여 허용 가능한 범위를 더 넓힐 것
        # 예: max_P = 20 → max_P = 50로 조정
        # P[0,0], P[1,1] 값이 적절하다면: 다른 원인을 점검해야 하며, 상태 변경 조건 (confirmed_threshold, delete_threshold)을 다시 살펴보아야 함
        print(f"Track ID {track.id} - Score: {track.score}, State: {track.state}, P[0,0]: {track.P[0,0]}, P[1,1]: {track.P[1,1]}")

        if track.score >= params.delete_threshold:
            track.state = 'tentative'  # 그렇지 않으면 '잠정적' 상태
            print(f"Track ID {track.id} is now tentative") #디버그코드
        if track.score >= params.confirmed_threshold:
            track.state = 'confirmed'  # 점수가 임계값을 넘으면 '확정' 상태로 전환
            print(f"Track ID {track.id} is now confirmed") #디버그코드
        #여기까지 멘토가 준 코드 (아래의 기존코드 대체)
        

        ''' 
        #기존 코드
        track.score += 1  # 트랙 점수 증가
        if track.score > params.confirmed_threshold:
            track.state = 'confirmed'  # 점수가 임계값을 넘으면 '확정' 상태로 전환
        else:
            track.state = 'tentative'  # 그렇지 않으면 '잠정적' 상태
        '''

        '''
        chatgpt 조언 : 
        트랙 점수 증가 방식:

        첫 번째 코드(멘토의 코드)에서는 track.score가 1. / params.window만큼 증가합니다. 
        이는 트랙 점수 증가에 가중치를 두는 방식으로, params.window의 값에 따라 점수가 더 천천히 또는 더 빠르게 증가할 수 있습니다.
        두 번째 코드(기존코드)에서는 track.score가 1씩 고정된 값으로 증가합니다. 이는 더 단순한 증가 방식입니다.
        
        상태 전환 조건:

        첫 번째 코드(멘토 코드)에서는 두 개의 조건문이 사용되어, track.score가 params.delete_threshold 이상일 때 상태를 'tentative'로 설정하고, 
        params.confirmed_threshold 이상일 때 'confirmed'로 전환합니다. 즉, 두 가지 조건을 순차적으로 확인하여 트랙 상태를 변경합니다.
        두 번째 코드(기존코드) 에서는 track.score가 params.confirmed_threshold를 초과할 때 'confirmed' 상태로 전환되고, 
        그렇지 않은 경우 'tentative'로 설정합니다. 여기서는 단일 조건에 의해 상태가 결정됩니다.
        주요 차이점 요약:

        첫 번째 코드(멘토코드)는 트랙 점수 증가 방식이 가중치를 사용하고, 상태 전환이 두 가지 조건에 따라 이루어집니다.
        두 번째 코(기존코드)드는 점수가 1씩 증가하며, 단일 조건을 기준으로 상태가 전환됩니다.
        따라서 첫 번째 코드는 보다 정교한 트랙 관리 및 점수 증가 로직을 제공하는 반면, 두 번째 코드는 더 단순한 로직입니다
        '''


        #pass
        
        ############
        # END student code
        ############ 