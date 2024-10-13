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
#이 함수는 LiDAR 데이터의 범위(range)와 강도(intensity) 이미지를 시각적으로 표현하는 함수
def show_range_image(frame, lidar_name): #show_range_image 함수,역할: LiDAR 범위(range) 데이터를 시각화하여 각 포인트의 범위와 강도(intensity)를 보여줌.
    # frame: Waymo 데이터셋의 프레임 객체 ( LiDAR 데이터를 포함하는 정보)
    # idar_name: 처리할 LiDAR 센서의 이름 (예를 들어, 'TOP', 'FRONT' 등 특정 LiDAR 센서를 선택)

    ####### ID_S1_EX1 START #######   라이다 데이터를 시각화
    print("show_range_image - student task")

    # extract lidar data and range image for the roof-mounted lidar
    #LiDAR 데이터 추출 및 압축 해제:
    lidar = waymo_utils.get(frame.lasers, lidar_name)
           #waymo_utils.get: frame 객체 안의 lasers 속성에서 lidar_name에 해당하는 LiDAR 데이터를 가져옴

    ri = dataset_pb2.MatrixFloat() #dataset_pb2.MatrixFloat(): Waymo 데이터 구조의 범위 이미지 데이터를 저장하는 객체.
    
    ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))  # Decompress and parse
    #ri.ParseFromString: 압축된 LiDAR 데이터를 풀어서 범위 이미지로 변환
    #내부로직 : lidar.ri_return1.range_image_compressed: LiDAR 데이터를 압축된 형태에서 복원하여, 범위 및 강도 값을 추출
    #zlib.decompress: 압축을 풀기 위한 함수로, 압축된 range_image_compressed 데이터를 복원

    ri = np.array(ri.data).reshape(ri.shape.dims)  # Reshape range image
        #np.array(ri.data): ri 객체의 데이터를 NumPy 배열로 변환
        #reshape: 배열의 차원을 재구성하여 이미지 형태로 만듬. ri.shape.dims는 이미지의 크기와 차원임


    # extract the range and the intensity channel from the range image
    #범위 및 강도 채널 추출:
    ri_range = ri[:, :, 0]  # range channel
    #ri[:, :, 0]: LiDAR 데이터의 첫 번째 채널을 사용해 범위 데이터를 추출

    ri_intensity = ri[:, :, 1]  # intensity channel
    #ri[:, :, 1]: 두 번째 채널에서 강도 데이터를 추출

    # set values <0 to zero
    #음수값을 0으로 설정
    ri_range[ri_range < 0] = 0.0 #ri_range[ri_range < 0] = 0.0: 범위 데이터 중 0보다 작은 값을 모두 0으로 만듬
    ri_intensity[ri_intensity < 0] = 0.0 #ri_intensity[ri_intensity < 0] = 0.0: 강도 데이터 중 0보다 작은 값을 모두 0으로 설정

    # map the range channel onto an 8-bit scale and normalize
    #범위 및 강도 데이터의 8비트 스케일 변환:
    ri_range = (ri_range - np.min(ri_range)) / (np.max(ri_range) - np.min(ri_range)) * 255
    img_range = ri_range.astype(np.uint8)
    #np.min(ri_range): 범위 데이터의 최소 값을 가져옴
    #np.max(ri_range): 범위 데이터의 최대 값을 가져옴
    #(ri_range - np.min(ri_range)) / (np.max(ri_range) - np.min(ri_range)) * 255: 범위 데이터를 0~255 사이로 정규화하여 8비트 이미지로 변환
    #astype(np.uint8): 정규화된 값을 8비트 정수로 변환하여 이미지로 만듬

    # map the intensity channel onto an 8-bit scale
    #강도 데이터의 8비트 스케일 변환:
    ri_intensity = np.clip(ri_intensity, 0, 1) * 255
                 #np.clip(ri_intensity, 0, 1): 강도 값을 0과 1 사이로 제한
                  
    img_intensity = ri_intensity.astype(np.uint8)
                 #astype(np.uint8): 강도 데이터를 8비트 이미지로 변환
    
    # stack the range and intensity image vertically
    #범위와 강도 이미지를 수직으로 쌓기:
    img_range_intensity = np.vstack((img_range, img_intensity))
                         #np.vstack((img_range, img_intensity)): 범위 이미지(img_range)와 강도 이미지(img_intensity)를 수직으로 쌓아 하나의 이미지로 만듬
    
    
    
    #내부로직 : img_range, img_intensity: LiDAR 데이터에서 범위와 강도를 시각적으로 표현하기 위해 8비트 이미지로 변환한 값
    return img_range_intensity #return img_range_intensity: 결합된 범위 및 강도 이미지를 반환

    ####### ID_S1_EX1 END #######


# visualize lidar point-cloud
# LiDAR 포인트 클라우드를 시각화하는 함수
# 이 함수는 Open3D 라이브러리를 사용해 LiDAR 포인트 클라우드를 시각화하는 함수입
def show_pcl(pcl):
    #pcl: LiDAR 포인트 클라우드 데이터를 입력으로 받는 변수. 이 데이터는 3차원 좌표로 이루어진 포인트들의 집합 
    

    ####### ID_S1_EX2 START #######  라이다 포인트 클라우드 시각화
    print("show_pcl - student task")

    # step 1: initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    #o3d.visualization.VisualizerWithKeyCallback(): Open3D 라이브러리에서 제공하는 시각화 객체를 생성. 이 객체는 사용자 입력에 따라 키보드 콜백을 받을 수 있음

    vis.create_window()
    #vis.create_window(): 시각화를 위한 창을 생성. 이 창에서 포인트 클라우드를 렌더링하여 보여줄 수 있음

    # step 2: create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
          #o3d.geometry.PointCloud(): Open3D에서 제공하는 포인트 클라우드 객체를 생성합니다. 이 객체는 3차원 좌표 데이터를 포함하고, 시각화를 위한 다양한 기능을 제공

    #내부 로직 : o3d.geometry.PointCloud: Open3D 라이브러리를 사용해 3D 포인트 클라우드를 시각화

    # step 3: set points in pcd instance by converting the point-cloud into 3d vectors
    #포인트 클라우드 데이터를 3D 벡터로 변환하여 설정:
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
    #pcl[:, :3]: 입력된 LiDAR 포인트 클라우드 데이터에서 첫 세 개의 열(x, y, z 좌표)을 가져옴
    #o3d.utility.Vector3dVector: NumPy 배열 형식의 3D 좌표 데이터를 Open3D의 3D 벡터 형식으로 변환
    #pcd.points: 생성된 포인트 클라우드 객체에 3D 좌표 데이터를 설정



    # step 4: add the pcd instance to visualization and update geometry
   #포인트 클라우드를 시각화에 추가하고 업데이트
    #내부로직 : vis: 시각화를 위한 Open3D의 시각화 객체로, 포인트 클라우드를 표시하고 창을 띄움
    vis.add_geometry(pcd)
    #vis.add_geometry(pcd): 포인트 클라우드 데이터를 시각화 객체에 추가하여 화면에 표시될 준비를 함

    vis.update_geometry(pcd)
    #vis.update_geometry(pcd): 포인트 클라우드 데이터를 업데이트하여 시각화에 반영

    # step 5: visualize point cloud and keep window open until right-arrow is pressed
    #vis.poll_events() #시각화 창에서 발생하는 사용자 이벤트(키 입력, 마우스 클릭 등)를 처리
    #vis.update_renderer() #vis.update_renderer(): 시각화 화면을 다시 그림
    #vis.run() #시각화 창을 실행하고, 사용자가 종료할 때까지 창을 유지
    #vis.destroy_window()

    ####### ID_S1_EX2 END #######



# create birds-eye view of lidar data  ,BEV 변환을 위한 포인트 클라우드 처리
#역할: 이 함수는 LiDAR 포인트 클라우드 데이터를 받아, 
# 이를 Bird's Eye View (BEV) 맵으로 변환하는 과정을 수행. 
# BEV는 위에서 본 모습을 시각화한 2D 이미지를 말합

#여기서부터 멘토가 주신 함수
#이 함수는 LiDAR 포인트 클라우드를 처리하여 강도, 높이, 밀도 정보를 가진 BEV 맵을 생성하고 이를 모델에 입력할 수 있는 형태로 변환함
def bev_from_pcl(lidar_pcl, configs):
    #lidar_pcl: 입력으로 들어오는 LiDAR 포인트 클라우드 데이터
    #configs: BEV 맵을 생성하기 위한 설정 값들이 포함된 객체
    
    # remove lidar points outside detection area and with too low reflectivity
        #포인트 필터링 ,mask: 설정된 범위 내에 있는 포인트만 필터링하는 마스크.
    
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    #mask: LiDAR 포인트 데이터 중에서 지정된 범위 내에 있는 포인트만 남기기 위해 마스크를 설정. 
    #이 마스크는 X, Y, Z 좌표가 주어진 설정(configs)의 한계 범위 내에 있을 때만 참(True)임.
    #lidar_pcl[mask]: 설정된 범위 내의 포인트만 추출하여 LiDAR 포인트 클라우드로 만듬




    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    #지면 레벨 조정
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]
    #lidar_pcl[:, 2]: LiDAR 데이터의 Z 좌표(높이)를 조정. 지면을 기준으로 설정된 높이를 빼줌으로써 BEV에서의 이미지 변화를 방지



    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######      BEV 변환을 위한 포인트 클라우드 처리
    print("bev_from_pcl - student task ID_S2_EX1")
    




    # step 1 : compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    #BEV맵으로 변환, bev_discretization: BEV 맵에서의 해상도를 설정하기 위해 X축 범위를 이미지 높이로 나누는 값.
    #bev_discretization: X축 범위를 BEV 이미지 높이로 나누어 BEV 맵의 해상도를 계산. 이 값은 X 좌표를 BEV 맵에서의 픽셀로 변환하는 데 사용



    # step 2 : create a copy of the lidar pcl and transform all metric x-coordinates into bev-image coordinates
    #BEV맵으로 변환, lidar_pcl_cpy[:, 0], lidar_pcl_cpy[:, 1]: 포인트 클라우드의 좌표를 BEV 맵 좌표로 변환.
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discretization))
    #np.copy(lidar_pcl): LiDAR 포인트 클라우드 데이터를 복사하여 변환에 사용할 복사본을 만듬
    #lidar_pcl_cpy[:, 0]: X 좌표를 BEV 맵의 픽셀 좌표로 변환


    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_cpy[:, 1] = np.int_(np.floor((lidar_pcl_cpy[:, 1] - configs.lim_y[0]) / bev_discretization))
    lidar_pcl_cpy[lidar_pcl_cpy[:, 1] < 0, 1] = 0  # avoid negative indices
    #lidar_pcl_cpy[:, 1]: Y 좌표도 같은 방식으로 변환하되, 음수 인덱스를 방지하기 위해 Y 좌표가 0보다 작으면 0으로 설정




    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    show_pcl(lidar_pcl_cpy) #섹션2 과제위해서
    ####### ID_S2_EX1 END #######
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######      BEV 맵의 강도(intensity) 레이어 계산
    print("student task ID_S2_EX2")
    
     ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    #강도 레이어 생성:    
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) #멘토가 수정한 것(+1추가)
    #intensity_map: BEV 맵의 강도(intensity) 레이어를 생성하기 위한 0으로 채워진 배열임. BEV 맵의 크기에 맞춰 생성


    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    #포인트 강도값 정렬 및 처리:
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0  #멘토가 준 것 (한줄 추가) ,# 강도 값을 1.0으로 제한
    #lidar_pcl_cpy[:,3]>1.0,3] = 1.0: 강도 값이 1을 넘는 포인트는 1로 클리핑하여 정규화
    index_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0])) #기존 [:, 2]였는데 멘토가 [, 3]으로 변경
                     #np.lexsort: X, Y 좌표 순서대로 포인트를 정렬하며, Z 좌표를 기준으로 정렬된 데이터를 가져옴
    lidar_pcl_top = lidar_pcl_cpy[index_intensity] #기존 [idx] -->멘토[index_intensity]
    





    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    #step3는 멘토가 완전히 바꿈
    lidar_num, lidar_indices, lidar_count = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_cpy[lidar_indices]
    #np.unique: X, Y 좌표가 중복되는 포인트 중 가장 위에 있는(Z가 가장 큰) 포인트만 남김
    



    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
   #step4도 멘토가 완전히 바꿈
   #강도 값 맵에 할당:
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3])-np.amin(lidar_pcl_top[:, 3]))
    #intensity_map: 강도 값을 BEV 맵의 픽셀에 할당. 이때 최대값과 최소값의 차이로 정규화하여 강도 값을 분포시키고, 차량 등의 객체가 뚜렷하게 보이도록 조정    


    # step 5: intensity map 시각화 (흑백 이미지로 시각화) 제프리의 코드를 따라함
    img_intensity = (intensity_map * 256).astype(np.uint8) #기존과 동일
    #cv2.imshow("Intensity map", img_intensity)  #섹션2를 위해서 비주석처리
    #cv2.waitKey(0) #멘토 주석처리
    #cv2.destroyAllWindows()  # #멘토 주석처리
    ####### ID_S2_EX2 END #######
    
    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     BEV 맵의 높이(height) 레이어 계산
    print("student task ID_S2_EX3")
    # step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1)) #멘토가 각각 +1추가
    #height_map: LiDAR 포인트의 높이를 저장할 배열을 생성하고, Z 값에 따라 정규화된 높이를 할당


    
    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map 
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0])) #기존과 동일
    
    #멘토가 주신 코드 (이미지 강도맵구현)
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    #cv2.imshow('height map', img_height) #섹션2 과제출력
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
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
    #밀도(density)레이어 계산 
    # density_map: 각 포인트 클라우드의 밀도를 계산하여 BEV 맵의 밀도 레이어를 생성
    #density_map: 포인트 밀도 정보를 저장할 배열을 생성합니다. 포인트의 수를 로그 스케일로 정규화하여 밀도 값을 할당



    # Use unique indices to get the counts correctly # ValueError shape mismatch방지목적으로 추가
    _, unique_indices, counts = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True, return_counts=True)
    
    # Normalize the counts # ValueError shape mismatch방지목적으로 추가
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    
    #밀도(density)레이어 계산 , normalizedCounts: 포인트의 밀도를 정규화하여 0~1 사이 값으로 표현
    # Assign normalized counts using unique indices # ValueError shape mismatch방지목적으로 추가
    density_map[np.int_(lidar_pcl_top[unique_indices, 0]), np.int_(lidar_pcl_top[unique_indices, 1])] = normalizedCounts
    
    # assemble 3-channel bev-map from individual maps
    #3-채널 BEV 맵 생성
    #bev_map: 각 레이어(강도, 높이, 밀도)를 결합하여 RGB형식의 3-채널 BEV 맵을 생성
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map
    
    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map


    #BEV 맵을 텐서로 변환:
    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    #3-채널 BEV 맵 생성 : torch.from_numpy(bev_maps): 생성된 BEV 맵을 PyTorch 텐서로 변환하여 모델 입력으로 사용할 수 있게 함
                        #torch.from_numpy(bev_maps): NumPy 배열을 PyTorch 텐서로 변환하여 모델에 입력으로 사용



    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    #input_bev_maps: 텐서를 GPU 또는 CPU로 전송하여 모델에 사용할 준비를 함
    
    return input_bev_maps



''' 
멘토가 제공한 코드와 원래 코드를 비교하여 차량 검출에 성공하게 된 이유는, 몇 가지 중요한 변경 사항이 포인트 클라우드 데이터를 처리하는 과정에서 발생했기 때문입니다. 
특히, BEV 변환과 포인트 클라우드 필터링 및 정규화 방식에서 개선된 부분이 차량을 더 정확하게 검출하는 데 큰 영향을 미쳤습니다.

주요 변경 사항 및 역할
1. 강도(intensity) 처리 방식 개선

기존 코드: python 코드 복사
    idx = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[idx]

멘토 코드:python 코드 복사
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0  # 멘토가 준 것 (한줄 추가)
    index_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0])) #기존 [:, 2]였는데 멘토가 [, 3]으로 변경
    lidar_pcl_top = lidar_pcl_cpy[index_intensity]

차이점 및 효과:
    강도값 클리핑: lidar_pcl_cpy[:, 3] > 1.0에서 강도값이 1.0을 넘지 않도록 클리핑(clipping)을 추가하였습니다. 
    이 작업은 강도 값이 너무 높은 값으로 인해 발생할 수 있는 아웃라이어(outlier)를 방지하는 데 중요한 역할을 합니다. 
    강도값이 1.0을 넘으면 BEV 맵에서 너무 높은 값으로 나타나 불균형한 표현을 할 수 있습니다.
    Z값에서 강도로 변경: 정렬 기준을 lidar_pcl_cpy[:, 2](높이값)에서 lidar_pcl_cpy[:, 3](강도값)으로 변경하여 강도를 기준으로 BEV 맵을 생성합니다. 
    이는 차량과 같은 객체의 강도를 더 명확하게 표현하여 정확한 검출이 가능하게 만듭니다.

2. 포인트 클라우드에서 중복 제거 및 고유 좌표 처리 방식 개선
기존 코드: python 코드 복사
    unique_indices = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True)[1]
    lidar_top_pcl = lidar_pcl_top[unique_indices]

멘토 코드:python 코드 복사
    lidar_num, lidar_indices, lidar_count = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_pcl_top = lidar_pcl_cpy[lidar_indices]
차이점 및 효과:
    멘토 코드는 np.unique를 사용하여 고유한 X, Y 좌표를 찾을 때 그 좌표의 개수도 함께 저장하는 방식을 도입했습니다. 이를 통해 포인트 클라우드에서 각 고유 좌표에 대한 밀도 정보도 정확하게 반영할 수 있습니다.
    이 변화는 BEV 맵의 밀도(density) 레이어 계산에서 중요한 역할을 하며, 차량과 같은 객체의 밀도를 정확하게 표현하여 검출 성능을 향상시킵니다.

3. 강도 및 밀도 값의 정규화 방식 개선
기존 코드: python 코드 복사
    intensity_map[inds[:, 0], inds[:, 1]] = lidar_pcl_top[:, 3] / (np.percentile(lidar_pcl_top[:, 3], 99) - np.percentile(lidar_pcl_top[:, 3], 1))

멘토 코드:python 코드 복사
    intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3])-np.amin(lidar_pcl_top[:, 3]))
차이점 및 효과:
    정규화 방식 변경: 기존 코드에서는 강도를 99번째와 1번째 백분위수 사이에서 정규화했지만, 멘토 코드는 최대값과 최소값을 이용해 정규화했습니다. 이는 값의 범위를 더 균일하게 만들어 아웃라이어에 덜 민감하게 작용할 수 있습니다.
    이 정규화 방식은 강도 및 밀도 표현에서 객체의 경계와 특징을 더 잘 구분할 수 있게 하여 검출 성능을 높이는 데 기여합니다.

4. 밀도 레이어 처리 개선
기존 코드: python 코드 복사
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts

멘토 코드: python 코드 복사
    density_map[np.int_(lidar_pcl_top[unique_indices, 0]), np.int_(lidar_pcl_top[unique_indices, 1])] = normalizedCounts
차이점 및 효과:
    멘토 코드는 밀도 레이어에서 중복 좌표를 처리하는 방식을 개선하여 중복된 포인트들이 정확하게 반영되도록 했습니다. 
    이를 통해 밀도가 더 정밀하게 계산되어 차량과 같은 객체를 더 잘 구분할 수 있습니다.
    이 개선은 BEV 맵의 밀도 표현을 더 정확하게 만들어 차량 검출 성능을 향상시킵니다.
    결론 : 멘토가 제공한 코드에서 강도와 밀도 처리 방식이 개선되었고, 포인트 클라우드에서 중복 좌표와 고유 좌표를 처리하는 방식이 변경되었습니다. 
    이러한 변화들은 BEV 맵의 강도와 밀도 레이어를 더 정확하게 계산하고 시각화하는 데 중요한 역할을 했으며, 이를 통해 차량 검출 성능이 개선되었습니다.






나의 말:
objdet_pcl_jeffery.py
너가 주었던 bev from pcl함수는 아마 너가 제프리 것을 모방한 것일텐데..제프리것을 업로드해줄테니 비교해바.. 너가 그것을 모방한건지..
ChatGPT의 말: ChatGPT
The differences between the two versions of the bev_from_pcl function primarily center on how data is handled, especially for intensity normalization, 
filtering, and sorting. Here's a comparison between your earlier code (likely from Jeffrey's original) and the mentor-upgraded version, 
along with an explanation of why it likely works better in the context of detecting cars in LiDAR images:

Key Differences and Why They Matter : Intensity Normalization and Filtering:

Original Code (Jeffrey's): 
    Intensity values are clipped to 1.0 (lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0), and intensity normalization is done using the difference between max and min values of the point cloud, i.e., intensity_map[inds[:, 0], inds[:, 1]] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3]) - np.amin(lidar_pcl_top[:, 3])). This direct approach could lead to cases where small outliers dominate the normalization range.
    Mentor's Updated Code: The intensity normalization uses percentiles to mitigate the influence of outliers: intensity_map[inds[:, 0], inds[:, 1]] = lidar_pcl_top[:, 3] / (np.percentile(lidar_pcl_top[:, 3], 99) - np.percentile(lidar_pcl_top[:, 3], 1)). This method is more robust as it reduces the impact of noise and extreme values, making objects like cars stand out more clearly.
    Effect: This improved normalization likely plays a crucial role in better identifying objects in the BEV, like cars, as it emphasizes the intensity contrast between relevant objects (e.g., cars) and irrelevant background noise.

Handling Unique Points and Sorting:

    Original Code: Sorting was performed based on -z (height) values, but the mentor changed this to intensity (lidar_pcl_cpy[:, 3]) for sorting in the intensity map step.
    Mentor's Updated Code: The mentor's code reorders points based on intensity before sorting and uses more precise handling of unique points in both intensity and height maps (np.unique function with return_counts=True). This ensures that only the most relevant topmost points are used for generating the final BEV maps.
    Effect: This adjustment prevents multiple points at the same x, y coordinates from skewing the map and ensures that the most relevant data (like the topmost point) is considered. This is especially important for accurately detecting cars, as their top surfaces are typically captured in LiDAR.

Visualization and Debugging Aids:

    The mentor’s code includes some additional debugging tools like visualizing intermediate maps (e.g., the intensity map) to ensure that the separation between cars and background is clear. This type of visualization can guide the debugging process, helping to see whether adjustments are needed.
    Effect: The addition of better visualization allows developers to directly assess the BEV map's quality, ensuring that the changes made lead to more accurate car detection in real-time.

Conclusion
    The mentor's updates have mainly improved robustness in data handling (especially intensity scaling and unique point selection), which directly impacts how well objects like cars are detected in BEV maps. These changes likely contribute to the improved performance in detecting cars in your BEV visualizations.

    
    
    
주요 차이점 및 그 이유
강도(intensity) 정규화 및 필터링:

제프리의 원래 코드: 
    강도 값이 1.0을 넘으면 클리핑하여 lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0로 처리하고, 강도 값을 맥스-민 차이로 정규화하여 intensity_map[inds[:, 0], inds[:, 1]] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3]) - np.amin(lidar_pcl_top[:, 3])) 방식으로 계산합니다. 이 방식은 데이터에서 극단값(노이즈)에 의해 정규화 범위가 크게 왜곡될 수 있습니다.
    멘토의 업데이트된 코드: 멘토는 정규화 시 상위 99백분위수와 하위 1백분위수 값을 사용하여 intensity_map[inds[:, 0], inds[:, 1]] = lidar_pcl_top[:, 3] / (np.percentile(lidar_pcl_top[:, 3], 99) - np.percentile(lidar_pcl_top[:, 3], 1))로 처리했습니다. 이 방식은 극단적인 값들의 영향을 줄여줍니다.

효과: 
    멘토의 방법은 노이즈나 극단값의 영향을 줄여서, 차량과 같은 객체가 배경과 명확히 구분될 수 있도록 강도 값을 더 적절하게 처리합니다. 즉, 차량을 더 잘 검출할 수 있게 해줍니다.

고유 포인트 처리 및 정렬:

    제프리의 원래 코드: 제프리의 코드는 Z(높이) 값을 기준으로 정렬한 후 고유한 포인트를 추출했습니다.
    멘토의 업데이트된 코드: 멘토는 Z 값 대신 강도 값을 기준으로 정렬(np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0])))하고, 고유 포인트 추출 시 np.unique 함수를 사용하여 반환된 인덱스와 카운트도 함께 추출합니다. 또한, 고유 포인트에 대한 처리를 더 세밀하게 진행해 더 정확한 결과를 얻게 합니다.
    효과: 이런 방식으로 강도 값이 더 중요한 기준이 되어, 중복된 좌표를 제거하고 가장 관련성이 높은 데이터를 사용합니다. 특히 차량처럼 눈에 띄는 물체의 상단을 잘 검출할 수 있게 도와줍니다.

시각화 및 디버깅 도구:

    제프리의 원래 코드: 기본적인 시각화만 포함되어 있습니다.
    멘토의 업데이트된 코드: 멘토의 코드에서는 중간 결과물(예: 강도 맵)을 시각화하여 더 나은 디버깅이 가능하게 합니다. 이를 통해 결과물이 어떻게 생성되는지를 직관적으로 확인할 수 있습니다.
    효과: 이 시각화를 통해 BEV 맵의 품질을 직접 확인할 수 있으며, 검출된 차량이 제대로 보이는지 즉시 판단할 수 있습니다.

결론
    멘토의 업데이트는 강도 정규화와 고유 포인트 선택 방식을 개선하여, 데이터 처리의 신뢰성을 높였습니다. 특히 차량과 같은 객체를 라이다 데이터에서 더 명확하게 구분하고 검출할 수 있게 되었습니다.
        
    
    '''

