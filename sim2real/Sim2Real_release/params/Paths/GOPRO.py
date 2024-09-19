import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)

GOPRO = ED()
GOPRO.train = ED()
GOPRO.train.dense_rgb = '/ailab/group/pjlab-sail/zhangziran/Syn_DATASET/GOPRO/RGB_dense/train_x8interpolated/'
GOPRO.train.event = "/ailab/group/pjlab-sail/zhangziran/workspace_sail/SynData4event/gray_event_10_new_fliter_2rd/train_x8interpolated/EVS"

GOPRO.test = ED()
GOPRO.test.dense_rgb = '/ailab/group/pjlab-sail/zhangziran/Syn_DATASET/GOPRO/RGB_dense/test_x8interpolated/'
GOPRO.test.event = "/ailab/group/pjlab-sail/zhangziran/workspace_sail/SynData4event/gray_event_10_new_fliter_2rd/test_x8interpolated/EVS"