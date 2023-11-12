#---------------------------------Set Dectron2-----------------------------------------------#
# refer to: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=7d3KxiHO_0gb
from __future__ import absolute_import, division, print_function
import torch, detectron2
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
# from google.colab.patches import cv2_imshow # this is for Google colab, we use cv2 saveimg instead.

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from mono_infer import parse_args, init_mono, test_simple, get_paths

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)


#---------------------------------Set Mono2-----------------------------------------------#
#refer to https://github.com/nianticlabs/monodepth2(I do not think it is a good model)


import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
args = parse_args()
# FINDING INPUT IMAGES
encoder, depth_decoder, feed_height, feed_width = init_mono(args, device)

#---------------------------------Set Overall--------------------------------------------#
import multiprocessing
#multiprocessing.set_start_method('spawn', force=True)

print_config = True
if print_config:
    #!nvcc --version
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
    print("detectron2:", detectron2.__version__)
    print(device)

image_path = '/hpc2hdd/home/wenshuozhang/wsZHANG/cardepth/monodepth2/assets/test_image.jpg'
paths, output_directory = get_paths(args, image_path)


#---------------------------------parallel_function---------------------------------------#
def mono_brunch(idx, image_path):
    # Load image and preprocess
    input_image = pil.open(image_path).convert('RGB')
    original_width, original_height = input_image.size
    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    # PREDICTION
    input_image = input_image.to(device)
    #def test_simple(encoder, depth_decoder, feed_height, feed_width, input_image, pred_metric_depth, idx, saving_flag=True):
    resized_depth = test_simple(encoder=encoder, depth_decoder=depth_decoder, input_image=input_image,original_width=original_width, original_height=original_height, pred_metric_depth=args.pred_metric_depth, idx=idx, output_directory=output_directory, saving_flag=False)
    return resized_depth

def dect_brunch(idx, im):
    im = im
    outputs = predictor(im)
    #print(outputs["instances"].pred_classes)
    #print(outputs["instances"].pred_boxes)
    # Car is label-2
    return outputs["instances"]

#-------------------------------------main_function---------------------------------------#
if __name__ == '__main__':
    #multiprocessing.set_start_method('spawn')
    for idx, image_path in enumerate(paths):
        if image_path.endswith("_disp.jpg"):
            # don't try to predict disparity for a disparity image!
            continue

        # we start a parallel
        # pool = multiprocessing.Pool(processes=2)
        # depth = pool.apply_async(mono_brunch(idx, image_path))
        # road_seg_location = pool.apply_async(dect_brunch(idx, image_path))
        # pool.close()
        # pool.join()
        im = cv2.imread(image_path)
        depth = mono_brunch(idx, image_path)
        road_seg_location = dect_brunch(idx, im)
        print("attention! look here: ", type(depth), depth.shape)
        print(road_seg_location.pred_classes)
        print(road_seg_location.pred_boxes)
        
        disp_resized_np = depth.squeeze().cpu().numpy()

        pred_all_class = road_seg_location.pred_classes.cpu().numpy()
        pred_all_box = road_seg_location.pred_boxes

        
        index = 0
        for box in pred_all_box:
            if pred_all_class[index] == 2:
                x1, y1, x2, y2 = map(int, box)
                car_depth = np.mean(disp_resized_np[y1:y2, x1:x2])
                text = f"Depth: {car_depth:.2f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 1
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

                im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                im = cv2.putText(im, text, (x1, y2 - text_size[1]), font, font_scale, (0, 0, 255), thickness)
            index+=1
        cv2.imwrite("/hpc2hdd/home/wenshuozhang/wsZHANG/cardepth/monodepth2/assets/test_car_dis.png", im)