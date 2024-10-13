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


# general package imports
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2
from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.resnet.models import fpn_resnet
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
# load model-related parameters into an edict
def load_configs_model(model_name='darknet', configs=None):

    # init config file, if none has been passed
    if configs == None:
        configs = edict()

    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))

    # set parameters according to model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet') #워크스페이스 드라이브경로 tools/objdet_models/darknet
                                #configs.model_path: 모델의 경로를 설정합니다. 여기서는 darknet 모델 파일이 저장된 경로를 지정
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
                                    #configs.pretrained_filename: 미리 학습된 darknet 모델의 가중치 파일 경로
                                    #model_path에서 정의된 위치를 가져와서 darknet 하위에 pretrained경로로 이동하여 파일 로딩
        configs.arch = 'darknet' #configs.arch: 모델 아키텍처를 나타내는 변수로, 이 경우 darknet을 사용
        configs.batch_size = 4 #configs.batch_size: 학습이나 추론 시 한 번에 처리할 데이터의 수. 여기서는 4로 설정
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
                          ##configs.cfgfile: darknet 모델의 설정 파일 경로
        configs.conf_thresh = 0.5 #configs.conf_thresh: 객체 감지 시 확신 임계값(Confidence Threshold).확신도가 이 값보다 높은 객체만 감지
        configs.distributed = False #configs.distributed: 모델을 분산 학습할지 여부를 설정.여기서는 False로 설정
        configs.img_size = 608 #configs.img_size: 입력 이미지의 크기.608x608 크기의 이미지를 사용
        configs.nms_thresh = 0.4 #configs.nms_thresh: 비최대 억제(Non-Maximum Suppression)에서 사용하는 임계값
        configs.num_samples = None #configs.num_samples: 샘플의 개수를 지정하는 변수.여기서는 설정되지 않아 None으로 설정
        configs.num_workers = 4 #configs.num_workers: 데이터를 로드하는 동안 사용할 워커의 수. 4로 설정
        configs.pin_memory = True #configs.pin_memory: GPU 학습 시 데이터를 고정 메모리에 올려 성능을 최적화할지 여부를 결정
        configs.use_giou_loss = False #configs.use_giou_loss: GIoU 손실을 사용할지 여부를 설정하는 플래그

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######
        #######
        print("student task ID_S3_EX1-3")
        #configs.saved_fn = 'fpn_resnet_18'
        #configs.arch = 'fpn_resnet_18'
        configs.saved_fn = 'fpn_resnet' #configs.saved_fn: 저장 파일 이름을 지정하는 변수.여기서는 'fpn_resnet'으로 설정
        configs.arch = 'fpn_resnet'  #configs.arch: 아키텍처 이름을 설정하는 변수.여기서는 'fpn_resnet' 아키텍처를 사용
        configs.model_path = configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'resnet')
                             #configs.model_path: fpn_resnet 모델 파일 경로
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'fpn_resnet_18_epoch_300.pth')
                                      #configs.pretrained_filename: 미리 학습된 fpn_resnet 모델의 가중치 파일 경로
                                      
                                    
        configs.K = 50 #configs.K: 한 번에 탐지할 객체의 최대 수를 설정하는 변수
        configs.no_cuda = True #configs.no_cuda: CUDA를 사용할지 여부를 설정하는 변수로, True면 CPU를 사용하고, False면 GPU를 사용
        configs.gpu_idx = 0 #configs.gpu_idx: GPU를 사용할 때, 사용할 GPU의 인덱스
        configs.num_workers = 1 #configs.num_workers: 데이터 로딩 시 사용할 워커의 수
        configs.batch_size = 1 #configs.batch_size: 학습이나 추론 시 한 번에 처리할 데이터의 수. 여기서는 1로 설정
        configs.conf_thresh = 0.5 #configs.conf_thresh: 객체 감지 시 확신 임계값
        configs.peak_thresh = 0.2 #configs.peak_thresh: 피크 검출을 위한 임계값
        configs.save_test_output = False #configs.save_test_output: 테스트 결과를 저장할지 여부를 설정하는 플래그
        configs.output_format = 'image' #configs.output_format: 출력 형식. 여기서는 'image'로 설정
        configs.output_video_fn = 'out_fpn_resnet_18' #configs.output_video_fn: 출력 비디오 파일 이름
        configs.output_width = 608 #configs.output_width: 결과 이미지의 너비

        configs.pin_memory = True #configs.pin_memory: GPU 학습을 위한 고정 메모리 사용 여부를 설정
        configs.distributed = False  # For testing on 1 GPU only

        configs.input_size = (608, 608) #configs.input_size: 입력 이미지의 크기
        configs.hm_size = (152, 152) #configs.hm_size: 히트맵 크기
        configs.down_ratio = 4 #configs.down_ratio: 다운샘플링 비율
        configs.max_objects = 50 #configs.max_objects: 한 장의 이미지에서 최대 탐지할 객체 수

        configs.imagenet_pretrained = False #configs.imagenet_pretrained: ImageNet 사전 학습 모델을 사용할지 여부
        configs.head_conv = 64 #configs.head_conv: 컨볼루션 레이어의 필터 개수
        configs.num_classes = 3 #configs.num_classes: 탐지할 객체의 클래스 수
        configs.num_center_offset = 2 #configs.num_center_offset: 중심 오프셋의 수
        configs.num_z = 1 #configs.num_z: Z 좌표의 수
        configs.num_dim = 3 #configs.num_dim: 차원의 수
        configs.num_direction = 2  # sin, cos #configs.num_direction: 방향의 수

        configs.heads = { #각 출력 헤드를 설정하는 변수
            'hm_cen': configs.num_classes,
            'cen_offset': configs.num_center_offset,
            'direction': configs.num_direction,
            'z_coor': configs.num_z,
            'dim': configs.num_dim
        }
        configs.num_input_features = 4 #configs.num_input_features: 입력 특징의 수

        ####################################################################
        ############## Dataset, Checkpoints, and results dir configs #########
        ####################################################################
        configs.root_dir = '../'

        #######
        ####### ID_S3_EX1-3 END #######

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True  # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))

    return configs


# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet', configs=None):
    # Configs=None --> 두 번째 매개변수입니다. 기본값은 None이며, 이미 설정된 구성(configs)이 있으면 전달할 수 있음. 없으면 새로운 설정을 생성

    # init config file, if none has been passed
    if configs == None: #configs가 None인 경우 (즉, 함수 호출 시 별도로 설정을 전달하지 않았을 때) 새로운 설정을 생성하라는 조건문
        configs = edict() #configs라는 변수에 빈 edict(쉽게 말해, 확장 가능한 딕셔너리)를 생성하여 할당
                          #edict는 딕셔너리와 비슷하지만 속성처럼 접근할 수 있는 확장형 딕셔너리                                                                                   
    # birds-eye view (bev) parameters
    configs.lim_x = [0, 50]  # detection range in m, BEV(조감도) 맵의 x축 감지 범위를 0m에서 50m로 설정
    configs.lim_y = [-25, 25] #y축 감지 범위를 -25m에서 25m로 설정
    configs.lim_z = [-1, 3]  #z축 감지 범위를 -1m에서 3m로 설정. 이는 라이다가 감지하는 높이 범위
    configs.lim_r = [0, 1.0]  # reflected lidar intensity,라이다의 반사 강도(reflectivity)를 0에서 1까지 설정
    configs.bev_width = 608  # pixel resolution of bev image ,BEV 맵의 가로 해상도를 608 픽셀로 설정
    configs.bev_height = 608 #BEV 맵의 세로 해상도를 608 픽셀로 설정
    configs.min_iou=0.5 #감지된 물체와 실제 물체의 IOU(Intersection Over Union)의 최소값을 0.5로 설정. 0.5 이상인 경우 물체 감지가 성공했다고 판단

    # add model-dependent parameters
    configs = load_configs_model(model_name, configs) #이 함수는 모델 이름에 따라 추가적으로 필요한 설정을 불러옴. 예를 들어, 'fpn_resnet'이라는 모델의 경우 그에 맞는 설정이 추가

    # visualization parameters
    configs.output_width = 608  # width of result image (height may vary), 결과 이미지의 가로 크기를 608 픽셀로 설정 (결과 이미지 세로크기는 다를 수 있음)
    # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
                         #[0, 255, 255]: 첫 번째 객체(예: 보행자) 색상: 노란색.
                         #[0, 0, 255]: 두 번째 객체(예: 차량) 색상: 빨간색
                         #[255, 0, 0]: 세 번째 객체(예: 자전거 이용자) 색상: 파란색.
    return configs #모든 설정 값을 담은 configs 변수를 반환




###################################
#직접 레이어수 설정하여 ValueError: invalid literal for int() 에러 방지

# create model according to selected model type
# create_model: 이 함수는 주어진 설정에 따라 모델을 생성하는 역할
#위의 함수에서 return된 configs(물체감지 설정값 담고 있음)을 받는 매개변수
def create_model(configs):

    # check for availability of model file
    # assert : 조건이 참인지 확인, 조건이 거짓이면 오류 발생시킴
    #os.path.isfile(configs.pretrained_filename): 설정(configs)에 포함된 사전 학습된 모델 파일이 존재하는지 확인. 파일이 없으면 오류가 발생
    #"No file at {}".format(configs.pretrained_filename): 파일이 없을 경우 출력될 에러 메시지. {}는 파일 경로를 넣는 자리로, configs.pretrained_filename 값이 출력
    assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)

    # create model depending on architecture name
    #(configs.cfgfile is not None): 또한, configs에서 'darknet' 설정 파일(cfgfile)이 존재하는지 확인
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
                #'darknet' 모델을 생성. cfgfile은 설정 파일 경로이고, use_giou_loss는 GIoU 손실을 사용할지 여부를 설정
        
    elif 'fpn_resnet' in configs.arch:
        #'fpn_resnet' in configs.arch: 모델 아키텍처에 'fpn_resnet'이 포함된 경우 해당 조건이 참
        print('using ResNet architecture with feature pyramid')
        
        ####### ID_S3_EX1-4 START #######
        #######
        print("student task ID_S3_EX1-4")

        
        layers = 18 # 직접 레이어 수를 설정
        model = fpn_resnet.get_pose_net(num_layers=layers, heads=configs.heads, head_conv=configs.head_conv, imagenet_pretrained=configs.imagenet_pretrained)
        #num_layers=layers: 레이어 수를 설정
        #heads=configs.heads: 모델의 출력 헤드를 설정
        #head_conv=configs.head_conv: convolution 레이어의 필터 수를 설정
        #imagenet_pretrained=configs.imagenet_pretrained: ImageNet에서 사전 학습된 모델을 사용할지 여부를 설정
        ####### ID_S3_EX1-4 END #######

    else: #else: 위의 조건들이 모두 거짓일 경우 실행
        assert False, 'Undefined model backbone' 
        #assert False: 오류를 발생시키는 코드
        #Undefined model backbone': 지원하지 않는 모델 아키텍처일 경우 발생하는 오류 메시지

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    #model.load_state_dict: 사전 학습된 모델의 가중치(파라미터)를 로드
    
    torch.load(configs.pretrained_filename, map_location='cpu')

    #configs.pretrained_filename에서 모델의 가중치를 불러옴 
    #map_location='cpu'는 가중치를 CPU에서 로드한다는 의미입
    
    print('Loaded weights from {}\n'.format(configs.pretrained_filename))
    #전 학습된 모델의 가중치를 성공적으로 불러왔을 때 그 파일 경로를 출력

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    #configs.device = torch.device: 모델이 학습 또는 추론에 사용할 장치(CPU 또는 GPU)를 설정
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    #model = model.to(device=configs.device): 모델을 위에서 설정한 장치(CPU 또는 GPU)에 로드
   
   
   
    model.eval() #model.eval(): 모델을 평가 모드로 전환. \
                 #이 모드에서는 배치 정규화 및 드롭아웃과 같은 레이어가 비활성화

    return model #생성한 모델 반환


##############################




# detect trained objects in birds-eye view
#함수가 Bird's Eye View(BEV) 이미지에서 훈련된 객체를 감지하는 것
def detect_objects(input_bev_maps, model, configs):

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():
        #with torch.no_grad(): 파이토치에서 autograd 엔진을 비활성화.이는 메모리 사용량을 줄이고, 추론(inference) 중 속도를 높이기 위한 목적으로 사용
        #input_bev_maps: 입력 매개변수, BEV 형태의 LiDAR 데이터
        #model: 사용 중인 객체 감지 모델
        #configs: 설정 값들이 포함된 객체로, 감지할 때 필요한 여러 파라미터가 들어 있음

        # perform inference
        outputs = model(input_bev_maps)
        #model(input_bev_maps): 모델에 입력 BEV 맵을 전달하여 추론을 수행 
                                        
        # decode model output into target object format
        if 'darknet' in configs.arch:
            #if 'darknet' in configs.arch: 모델 아키텍처가 'darknet'일 경우 조건문을 실행. configs.arch는 모델의 아키텍처 정보를 담고 있음
                        

            # perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) # 여기는 그대로 유지
                        #post_processing_v2: 객체 감지 결과를 후처리하는 함수. Non-Maximum Suppression(NMS)과 같은 후처리를 수행
                        #conf_thresh: 감지된 객체의 신뢰도 임계값
                        #nms_thresh: NMS에서 사용하는 임계값
            detections = [] #감지된 객체를 저장할 리스트
            for sample_i in range(len(output_post)): #output_post 리스트의 각 항목에 대해 반복문을 돌림
                                                     ##sample_i: 인덱스 번호
                if output_post[sample_i] is None:
                    #if output_post[sample_i] is None: 감지 결과가 없을 경우, 다음 루프로 
                    continue #ontinue: 현재 루프를 건너뛰고, 다음 루프
                detection = output_post[sample_i]  #detection: output_post에서 감지된 객체 정보를 가져옴
                for obj in detection: #for obj in detection: 감지된 객체들의 각 항목에 대해 반복문을 실행
                    x, y, w, l, im, re, _, _, _ = obj #obj: 하나의 감지된 객체
                    #x, y: 감지된 객체의 중심 좌표
                    #w, l: 감지된 객체의 너비와 길이
                    #im, re: 물체의 방향을 나타내는 복소수 값
                    #im, re: 물체의 방향을 나타내는 복소수 값
                    
                    yaw = np.arctan2(im, re)
                    #yaw: 감지된 객체의 방향을 나타내는 값. im과 re를 사용하여 방향을 계산
                    #np.arctan2: 두 값의 아크탄젠트(atan2)를 계산하여 방향을 결정
                    
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])
                    #detections.append: 감지된 객체의 정보를 리스트에 추가. 
                    #1은 감지된 객체의 클래스(예: 자동차)를 나타내며, x, y는 좌표, w, l은 너비와 길이, 
                    # yaw는 방향을 의미
            


        elif 'fpn_resnet' in configs.arch:
            #elif 'fpn_resnet' in configs.arch: 모델 아키텍처가 'fpn_resnet'일 경우, 이 조건문이 실행

            ####### ID_S3_EX1-5 START #######
            #######
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            
            #detections: decode 함수를 사용하여 모델 출력(outputs)을 해석한 결과를 저장
            #outputs['hm_cen'], outputs['cen_offset'] 등: 모델이 예측한 여러 값들로, 각각 객체의 중심점, 오프셋, 방향, z좌표, 크기 등을 의미
            #K=configs.K: 상위 K개의 객체를 선택하는 값
            
            detections = detections.cpu().numpy().astype(np.float32) 
            #이 부분에서 decoded_output이 Tensor에서 Numpy 배열로 변환
            #detections.cpu(): 텐서를 CPU 메모리로 이동시킴
            #.numpy(): 텐서를 NumPy 배열로 변환
            #.astype(np.float32): 배열의 데이터를 float32 형으로 변환

            #detections = post_processing(detections, configs) # 여기는 post_processing 함수 호출 시 conf_thresh와 같은 개별 인자를 전달하는 대신 configs 객체를 통째로 넘기는 방식
            #detections = detections[0][1]


            output_post = post_processing(detections, configs) # 여기는 post_processing 함수 호출 시 conf_thresh와 같은 개별 인자를 전달하는 대신 configs 객체를 통째로 넘기는 방식
            #output_post: 후처리된 결과를 저장하는 변수
            #post_processing(detections, configs): 감지된 객체에 대해 후처리를 수행하는 함수

            output_post = output_post[0][1]
            #output_post[0][1]: 후처리된 객체 중 특정 항목을 선택




            print(type(output_post))  # 결과값의 타입을 출력합니다.
            print(output_post)  # 결과값을 출력합니다.
            detections = output_post
            print("student task ID_S3_EX1-5")
                 


            '''
            #멘토의 코드 ,결과출력(텍스트)
            output_post = post_processing(decoded_output, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            output_post = output_post[0][1]
            print(type(output_post))
            print(output_post)
            '''

            #######
            ####### ID_S3_EX1-5 END #######

    ####### ID_S3_EX2 START #######
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = [] #objects: 최종 감지된 객체를 저장할 리스트

    #for det in detections: 감지된 객체들에 대해 반복문을 실행
    for det in detections:
        # step 1 : check whether there are any detections
        if len(det) > 0:
                #if len(det) > 0: 감지된 객체가 하나라도 있을 경우, 조건을 만족

            # step 2 : loop over all detections
                _, bev_x, bev_y, z, h, bev_w, bev_l, yaw = det
                #bev_x, bev_y: 감지된 객체의 BEV 좌표
                #z: 객체의 z 좌표
                #h: 객체의 높이
                #bev_w, bev_l: 객체의 너비와 길이
                #yaw: 객체의 방향



                # step 3 : perform the conversion using the limits for x, y and z set in the configs structure
                x = bev_y / configs.bev_height * \
                    (configs.lim_x[1] - configs.lim_x[0])
                #x = bev_y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])               
                
                
                y = bev_x / configs.bev_width * \
                    (configs.lim_y[1] - configs.lim_y[0]) + configs.lim_y[0]
                #y = bev_x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0]) + configs.lim_y[0]
                
                #x, y: BEV 좌표를 실제 월드 좌표로 변환하는 계산
                #configs.bev_height와 configs.bev_width를 사용해 비율을 맞춤
                
                
                
                
                z = z + configs.lim_z[0] #z: z 좌표에 제한 값을 추가
                
                w = bev_w / configs.bev_width * \
                    (configs.lim_y[1] - configs.lim_y[0])
                l = bev_l / configs.bev_height * \
                    (configs.lim_x[1] - configs.lim_x[0])
                #w, l: 객체의 너비와 길이를 실제 월드 좌표로 변환
                
                
                
                if ((x >= configs.lim_x[0]) and (x <= configs.lim_x[1])
                    and (y >= configs.lim_y[0]) and (y <= configs.lim_y[1])
                        and (z >= configs.lim_z[0]) and (z <= configs.lim_z[1])):
                # if ((x >= configs.lim_x[0]) and (x <= configs.lim_x[1]) and (y >= configs.lim_y[0]) and (y <= configs.lim_y[1]) and (z >= configs.lim_z[0]) and (z <= configs.lim_z[1])):
                # if: 감지된 객체가 설정된 범위 내에 있는지 확인하는 조건문. 
                # x, y, z 좌표가 각각 설정된 범위를 만족하는 경우에만 다음 코드를 실행     

                    # step 4 : append the current object to the 'objects' array
                    objects.append([1, x, y, z, h, w, l, yaw])
                    #objects.append: 객체 리스트에 감지된 객체를 추가. 이때 추가되는 값은 클래스, x, y, z 좌표, 높이, 너비, 길이, 그리고 yaw(방향)

    #######
    ####### ID_S3_EX2 END #######
    return objects


