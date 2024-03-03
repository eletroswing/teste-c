import cv2

import torch
import numpy as np
import os
import subprocess

from call.openpose.body import Body
from detectron2.config import get_cfg
import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.extractor import DensePoseResultExtractor

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file("./densepose_rcnn_R_50_FPN_s1x.yaml")
cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl"
predictor = DefaultPredictor(cfg)

num_classes = 18
input_size = [512, 512]

body_estimation = Body('./body_pose_model.pth')
state_dict = torch.load('./segmid.pth')['state_dict']

def getPose(cv2Img):
    candidate, subset = body_estimation(cv2Img)
    return candidate, subset

def getShcp(cv2Img):
    cv2.imwrite("input/model.jpg", cv2Img)
    command = f"python3 shcp/simple_extractor.py --dataset 'atr' --model-restore 'checkpoints/final.pth' --input-dir 'input' --output-dir 'out'"
    subprocess.call(command)
    img = cv2.imread("out/model.jpg", cv2.IMREAD_COLOR)
    return img

def getDensePose(cv2Img):
    outputs = predictor(cv2Img)['instances']

    densepose_results = DensePoseResultExtractor()(outputs)[0][0]
    i = densepose_results.labels.cpu().numpy()
    uv = densepose_results.uv.cpu().numpy() * 255
    iuv = np.stack((uv[1,:,:], uv[0,:,:], i * 0,))

    iuv = np.transpose(iuv, (1, 2, 0))

    return iuv


def GenerateThings(imgPth):
    with torch.no_grad():
        img = cv2.imread(imgPth, cv2.IMREAD_COLOR)
        print('get pose')
        pose = getPose(img)
        print('done, get densePose')
        densepose = getDensePose(img)
        print('done')
  
GenerateThings('/content/model.jpg')
