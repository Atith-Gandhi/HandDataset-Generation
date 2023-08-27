import os
import random

import cv2
import keyboard
import numpy as np
import open3d
import open3d as o3d
import pygame
from transforms3d.axangles import axangle2mat
import copy
import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils import *

def search_points(img_pth):
    img = cv2.imread(img_pth)
    a = 1


search_points("../nov29_imgs/generated_imgs/0_2.png")
