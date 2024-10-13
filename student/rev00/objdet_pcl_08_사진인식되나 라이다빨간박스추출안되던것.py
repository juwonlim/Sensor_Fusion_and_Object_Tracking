#이 코드는 LiDAR 데이터를 처리하여 Bird's Eye View (BEV) 맵을 생성하는 과정을 구현한 것임
# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools

import matplotlib.pyplot as plt   #bev맵에서 숫자출력위한 모듈

# visualize range image
def show_range_image(frame, lidar_name): #show_range_image 함수,역할: LiDAR 범위(range) 데이터를 시각화하여 각 포인트의 범위와 강도(intensity)를 보여줌.
    #인자 - frame: Waymo 데이터셋의 프레임 객체
    #인자   idar_name: 처리할 LiDAR 센서의 이름.

    ####### ID_S1_EX1 START #######   라이다 데이터를 시각화
    print("show_range_image - student task")

    # extract lidar data and range image for the roof-mounted lidar
    lidar = waymo_utils.get(frame.lasers, lidar_name)
    ri = dataset_pb2.MatrixFloat()
    ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))  # Decompress and parse
    #내부로직 : lidar.ri_return1.range_image_compressed: LiDAR 데이터를 압축된 형태에서 복원하여, 범위 및 강도 값을 추출
    
    ri = np.array(ri.data).reshape(ri.shape.dims)  # Reshape range image

    # extract the range and the intensity channel from the range image
    ri_range = ri[:, :, 0]  # range channel
    ri_intensity = ri[:, :, 1]  # intensity channel

    # set values <0 to zero
    ri_range[ri_range < 0] = 0.0
    ri_intensity[ri_intensity < 0] = 0.0

    # map the range channel onto an 8-bit scale and normalize
    ri_range = (ri_range - np.min(ri_range)) / (np.max(ri_range) - np.min(ri_range)) * 255
    img_range = ri_range.astype(np.uint8)

    # map the intensity channel onto an 8-bit scale
    ri_intensity = np.clip(ri_intensity, 0, 1) * 255
    img_intensity = ri_intensity.astype(np.uint8)

    # stack the range and intensity image vertically
    img_range_intensity = np.vstack((img_range, img_intensity))
    #내부로직 : img_range, img_intensity: LiDAR 데이터에서 범위와 강도를 시각적으로 표현하기 위해 8비트 이미지로 변환한 값
    return img_range_intensity

    ####### ID_S1_EX1 END #######


# visualize lidar point-cloud
#역할: LiDAR 포인트 클라우드를 시각화하는 함수
def show_pcl(pcl): 
    #pcl: LiDAR 포인트 클라우드 데이터

    ####### ID_S1_EX2 START #######  라이다 포인트 클라우드 시각화
    print("show_pcl - student task")

    # step 1: initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # step 2: create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
    #내부 로직 : o3d.geometry.PointCloud: Open3D 라이브러리를 사용해 3D 포인트 클라우드를 시각화

    # step 3: set points in pcd instance by converting the point-cloud into 3d vectors
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

    # step 4: add the pcd instance to visualization and update geometry
    #내부로직 : vis: 시각화를 위한 Open3D의 시각화 객체로, 포인트 클라우드를 표시하고 창을 띄움
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)

    # step 5: visualize point cloud and keep window open until right-arrow is pressed
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

    ####### ID_S1_EX2 END #######



# create birds-eye view of lidar data  ,BEV 변환을 위한 포인트 클라우드 처리
#역할: LiDAR 포인트 클라우드를 BEV로 변환하여 객체 탐지를 위한 입력 데이터를 만듬
def bev_from_pcl(lidar_pcl, configs):
    #lidar_pcl: LiDAR 포인트 클라우드 데이터.
    #configs: BEV 맵을 생성하기 위한 설정 값들

    # remove lidar points outside detection area and with too low reflectivity

    #포인트 필터링 ,mask: 설정된 범위 내에 있는 포인트만 필터링하는 마스크.
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######      BEV 변환을 위한 포인트 클라우드 처리
    print("bev_from_pcl - student task ID_S2_EX1")

    # step 1 : compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    #BEV맵으로 변환, bev_discretization: BEV 맵에서의 해상도를 설정하기 위해 X축 범위를 이미지 높이로 나누는 값.



    # step 2 : create a copy of the lidar pcl and transform all metric x-coordinates into bev-image coordinates
    #BEV맵으로 변환, lidar_pcl_cpy[:, 0], lidar_pcl_cpy[:, 1]: 포인트 클라우드의 좌표를 BEV 맵 좌표로 변환.
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discretization))

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_cpy[:, 1] = np.int_(np.floor((lidar_pcl_cpy[:, 1] - configs.lim_y[0]) / bev_discretization))
    lidar_pcl_cpy[lidar_pcl_cpy[:, 1] < 0, 1] = 0  # avoid negative indices

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    show_pcl(lidar_pcl_cpy)

    ####### ID_S2_EX1 END #######

    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######      BEV 맵의 강도(intensity) 레이어 계산
    print("student task ID_S2_EX2")

    # step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))
    #강도(intensity) 레이어 계산, intensity_map: 각 포인트의 강도를 사용해 BEV 맵의 강도 레이어를 생성.
    
    
    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    idx = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    #강도(intensity) 레이어 계산 , np.lexsort: X, Y, Z 순서로 정렬한 후 최상위 Z 좌표를 가져옴
    
    
    lidar_pcl_top = lidar_pcl_cpy[idx]

    # step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    unique_indices = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True)[1]
    #강도(intensity) 레이어 계산,unique_indices: 중복된 X, Y 좌표를 제거하여 고유한 좌표만 남김
    
    
    lidar_top_pcl = lidar_pcl_top[unique_indices]

    # step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0  # 강도 값 클리핑
    inds = lidar_pcl_top[:, :2].astype(np.int16)
    intensity_map[inds[:, 0], inds[:, 1]] = lidar_pcl_top[:, 3] / (np.percentile(lidar_pcl_top[:, 3], 99) - np.percentile(lidar_pcl_top[:, 3], 1))

    # step 5: intensity map 시각화 (흑백 이미지로 시각화) 제프리의 코드를 따라함
    img_intensity = (intensity_map * 256).astype(np.uint8)
    cv2.imshow("Intensity map", img_intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  # 이 코드를 추가하여 창이 닫히지(열리지?) 않는 문제 방지


    ####### ID_S2_EX2 END #######

    

    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     BEV 맵의 높이(height) 레이어 계산
    print("student task ID_S2_EX3")

    # step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height, configs.bev_width))
    #높이(height) 레이어 계산, height_map: 포인트의 높이 정보를 사용해 BEV 맵의 높이 레이어를 생성.




    # step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map
    height_map[np.int_(lidar_top_pcl[:, 0]), np.int_(lidar_top_pcl[:, 1])] = lidar_top_pcl[:, 2] / (configs.lim_z[1] - configs.lim_z[0])

    #멘토가 주신 코드 (이미지 강도맵구현)
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    cv2.imshow('height map', img_height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    ####### ID_S2_EX3 END #######

    #lidar_pcl_cpy = [] #여기서 리스트로 초기화됨, np(numpy)이므로 리스트로 초기화 되어선 안됨 -->chatgpt의 조언
    #lidar_pcl_top = []
    #height_map = []
    #intensity_map = []

    #chatgpt조언 데이터 타입 확인필요
    print(type(lidar_pcl_cpy))  # lidar_pcl_cpy의 데이터 타입을 출력
    print(lidar_pcl_cpy.shape)  # 배열의 크기를 출력


    # Compute density layer of the BEV map
    #density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) --> 기존 코드 
    
    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) #수정코드 (다음 에러방지목적 --> ValueError: shape mismatch: value array of shape (13446,) could not be broadcast to indexing result of shape (63745,))
    #밀도(density)레이어 계산 , density_map: 각 포인트 클라우드의 밀도를 계산하여 BEV 맵의 밀도 레이어를 생성


    # Use unique indices to get the counts correctly # ValueError shape mismatch방지목적으로 추가
    _, unique_indices, counts = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True)

    # Normalize the counts # ValueError shape mismatch방지목적으로 추가
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    #밀도(density)레이어 계산 , normalizedCounts: 포인트의 밀도를 정규화하여 0~1 사이 값으로 표현



    # Assign normalized counts using unique indices # ValueError shape mismatch방지목적으로 추가
    density_map[np.int_(lidar_pcl_top[unique_indices, 0]), np.int_(lidar_pcl_top[unique_indices, 1])] = normalizedCounts


    '''  ValueError: shape mismatch: value array of shape (13446,) could not be broadcast to indexing result of shape (63745,)) --> 이 에러방지위해 주석처리 하고 새로운 3개의 코드를 위에 삽입
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
    '''

    # assemble 3-channel bev-map from individual maps
    #3-채널 BEV 맵 생성
    #bev_map: 각 레이어(강도, 높이, 밀도)를 결합하여 3-채널 BEV 맵을 생성
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    #3-채널 BEV 맵 생성 : torch.from_numpy(bev_maps): 생성된 BEV 맵을 PyTorch 텐서로 변환하여 모델 입력으로 사용할 수 있게 함

    
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


