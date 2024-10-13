# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#


'''
#추가한 내용:
load_configs_model()에서 FPN-ResNet 모델에 대한 기본 설정을 추가
create_model()에서 ResNet 아키텍처를 사용해 모델을 생성하는 부분 구현
detect_objects()에서 후처리 결과에서 3D 경계 상자를 추출하는 논리를 구현
by lim
'''


# general package imports
import numpy as np
import torch
from easydict import EasyDict as edict

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing 

from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()  

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))    
    
    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False



    #elif model_name == 'fpn_resnet':
    ####### ID_S3_EX1-3 START #######
    #모델 이름에 따라 다른 모델 로드   
    elif model_name == 'fpn_resnet':
        print("Configuring FPN-ResNet Model")
        
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
        configs.arch = 'fpn_resnet'
        configs.batch_size = 2
        configs.conf_thresh = 0.4
        configs.nms_thresh = 0.5
        configs.num_samples = None
        configs.num_workers = 2
        configs.pin_memory = True

        # 추가된 heads 설정
        configs.heads = {
            'hm_cen': 3,          # Number of object classes (Pedestrian, Car, Cyclist)
            'cen_offset': 2,      # Center offset (x, y)
            'direction': 2,       # Direction (sin, cos)
            'z_coor': 1,          # Z-coordinate
            'dim': 3              # Dimension (width, length, height)
        }
        configs.head_conv = 64    # Convolutional head dimension
        configs.num_classes = 3   # <== 여기서 num_classes 속성을 추가 , 다음의 에러방지목적 -->AttributeError: EasyDict object has no attribute 'num classes
                                  #num_classes는 모델이 감지할 객체 클래스의 수를 지정하는 중요한 파라미터, 이것은 fpn_resnet과 같은 특정 아키텍처에서 객체 감지를 수행할 때 필요
                                    #따라서 darknet과 같은 다른 아키텍처에서는 반드시 필요하지 않을 수 있음. 그래서 fpn_resnet과 관련된 설정에만 추가함
         # **imagenet_pretrained 추가**
        configs.imagenet_pretrained = False  # 사전 학습된 모델을 사용할지 여부 (필요에 따라 True로 변경)

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU 설정
    configs.no_cuda = True
    configs.gpu_idx = 0
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs
####### ID_S3_EX1-3 END #######   






# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):

    # init config file, if none has been passed
    if configs==None:
        configs = edict()    

    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50] # detection range in m
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0] # reflected lidar intensity
    configs.bev_width = 608  # pixel resolution of bev image
    configs.bev_height = 608 

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs




#제프리 코드 참조한 함수
def create_model(configs):

    # 모델 파일의 경로가 올바른지 확인
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # 아키텍처 이름에 따라 모델 생성
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    

    elif 'fpn_resnet' in configs.arch:
        print('using ResNet architecture with feature pyramid')

        # 모델 생성 (제프리 방식 적용)
        model = fpn_resnet.get_pose_net(num_layers=18, heads=configs.heads, head_conv=configs.head_conv, imagenet_pretrained=configs.imagenet_pretrained)

    else:
        assert False, 'Undefined model backbone'

    # 모델 가중치 로드
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))

    # 모델을 평가 상태로 설정
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # 모델을 CPU 또는 GPU로 로드
    model.eval()          

    return model





# detect trained objects in birds-eye view
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # perform inference
        outputs = model(input_bev_maps) #모델의 추론을 담는 변수, 이 딕셔너리는 보통 모델의 출력값을 저장, 이 outputs는 모델이 예측한 여러 값을 딕셔너리 형태로 반환하는데, 여기에는 hm_cen, cen_offset, direction, z_coor, dim 같은 키가 포함되어 있어야 함 

        # decode model output into target object format
        if 'darknet' in configs.arch:

            # perform post-processing
            output_post = post_processing_v2(
                outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

        elif 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            ####### ID_S3_EX1-5 START #######
            # ResNet 모델로 추론한 후, 객체 검출 및 디코딩 수행.     
            #######
            print("student task ID_S3_EX1-5")
            #decoded_output = decode(outputs) #인자를 전달하지 못하여 에러발생 --> decoded_output = decode(outputs) , TypeError: decode() missing 4 required positional arguments: 'cen_offset', 'direction', 'z_coor', and 'dim' 
            decoded_output = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'], outputs['dim']) #이렇게 4개의 인자 전달하여 decode인자 전달못하는 에러방지,  K=configs.K이거는 여러후보중 상위 k개의 예측가져올때만 필요
            #output_post = post_processing(decoded_output, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) #수정전
            output_post = post_processing(decoded_output, configs) #수정후 --> 다음의 에러방지목적  ТуреЕггог: post_processing() got an unexpected keyword argument 'conf_thresh')
            detections = [] 
            for obj in output_post:
                x, y, w, l, h, yaw = obj
                detections.append([1, x, y, 0.0, h, w, l, yaw])
        
            #멘토가 주신 코드
            #Detect 부분 결과 출력하시기 위해서는 student task ID_S3_EX1-5의 output_post 결과값을 print로 출력해보시면 됩니다.
            #output_post = post_processing(decoded_output, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            output_post = output_post[0][1]
            print(type(output_post))
            print(output_post)

            #######
            ####### ID_S3_EX1-5 END #######     

            

    ####### ID_S3_EX2 START #######
    #디코딩된 결과로부터 3D 객체의 바운딩 박스를 추출하여 objects 배열에 저장
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] 
    for det in detections:
        if det is not None:
            # step 3: convert detection to 3D bounding box format
            _, x, y, z, h, w, l, yaw = det
            if configs.lim_x[0] <= x <= configs.lim_x[1] and \
               configs.lim_y[0] <= y <= configs.lim_y[1] and \
               configs.lim_z[0] <= z <= configs.lim_z[1]:
                objects.append([x, y, z, w, l, h, yaw])

    ## step 1 : check whether there are any detections

        ## step 2 : loop over all detections
        
            ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
        
            ## step 4 : append the current object to the 'objects' array
        
    #######
    ####### ID_S3_EX2 START #######   
    
    return objects    

