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



# visualize range image
def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######   라이다 데이터를 시각화  
    #######
    print("student task ID_S1_EX1")

    # step 1 : extract lidar data and range image for the roof-mounted lidar
    
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    range_image = dataset_pb2.MatrixFloat(lidar.ri_return1.range_image_compressed).values
    
    # step 2 : extract the range and the intensity channel from the range image
    
    range_channel = np.copy(range_image[:, :, 0])
    intensity_channel = np.copy(range_image[:, :, 1])
    
    # step 3 : set values <0 to zero
    
    range_channel[range_channel < 0] = 0
    intensity_channel[intensity_channel < 0] = 0
    
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    # 
    range_max = np.percentile(range_channel, 100)
    range_min = np.percentile(range_channel, 0)
    range_channel = 255 * (range_channel - range_min) / (range_max - range_min)
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    # 
    intensity_max = np.percentile(intensity_channel, 99)
    intensity_min = np.percentile(intensity_channel, 1)
    intensity_channel = 255 * (intensity_channel - intensity_min) / (intensity_max - intensity_min)


    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    # 
    img_range_intensity = np.vstack((range_channel, intensity_channel)).astype(np.uint8)
    img_range_intensity = [] # remove after implementing all steps
    #######
    ####### ID_S1_EX1 END #######     
    
    return img_range_intensity


# visualize lidar point-cloud
def show_pcl(pcl):

    ####### ID_S1_EX2 START #######  라이다 포인트 클라우드 시각화
      
    #######
    print("student task ID_S1_EX2")

    # step 1 : initialize open3d with key callback and create window
    # 
    import open3d as o3d
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    # step 2 : create instance of open3d point-cloud class
    # 
    pcd = o3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    # 
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    # 
    vis.add_geometry(pcd)
    
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    # 
    vis.run()
    vis.destroy_window()

    #######
   

    ####### ID_S1_EX2 END #######     
       


# create birds-eye view of lidar data  ,BEV 변환을 위한 포인트 클라우드 처리
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######      BEV 변환을 위한 포인트 클라우드 처리
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    ## 
    bev_discretization = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height

    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    ## 
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.floor(lidar_pcl_cpy[:, 0] / bev_discretization).astype(np.int32)    

    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    # 
    bev_discretization_y = (configs.lim_y[1] - configs.lim_y[0]) / configs.bev_width
    lidar_pcl_cpy[:, 1] = np.floor((lidar_pcl_cpy[:, 1] - configs.lim_y[0]) / bev_discretization_y).astype(np.int32)

    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # 
    show_pcl(lidar_pcl_cpy)
    #######
    ####### ID_S2_EX1 END #######     
    
    
    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######      BEV 맵의 강도(intensity) 레이어 계산
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    ## 
    intensity_map = np.zeros((configs.bev_height, configs.bev_width))

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    # 
    lidar_pcl_cpy = lidar_pcl_cpy[np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))]

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    ## 
    unique_pcl, indices, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    lidar_top_pcl = lidar_pcl_cpy[indices]

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    ## 
    intensity_map[np.int_(lidar_top_pcl[:, 0]), np.int_(lidar_top_pcl[:, 1])] = lidar_top_pcl[:, 3] / (np.amax(lidar_top_pcl[:, 3]) - np.amin(lidar_top_pcl[:, 3])) 
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    


    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    ## 
    cv2.imshow('Intensity Map', intensity_map)
    cv2.waitKey(0)

    #######
    ####### ID_S2_EX2 END ####### 


    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######     BEV 맵의 높이(height) 레이어 계산
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    ## 
    height_map = np.zeros((configs.bev_height, configs.bev_width))

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map
    ## 
    height_map[np.int_(lidar_top_pcl[:, 0]), np.int_(lidar_top_pcl[:, 1])] = lidar_top_pcl[:, 2] / (configs.lim_z[1] - configs.lim_z[0])

    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    ## 
    cv2.imshow('Height Map', height_map)
    cv2.waitKey(0)

    #######
    ####### ID_S2_EX3 END #######       

    # TODO remove after implementing all of the above steps
    lidar_pcl_cpy = []
    lidar_pcl_top = []
    height_map = []
    intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
        
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()
    return input_bev_maps


