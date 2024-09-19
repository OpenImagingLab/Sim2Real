import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED
import datetime


mkdir = lambda x:os.makedirs(x, exist_ok=True)


hostname = 'server' if 'PC' not in socket.gethostname() else 'local'

RC = ED()
RC.train = ED()
RC.train.rgb = "/ailab/user/zhangziran/Dataset/Sim2Real_release"
RC.train.evs = "/ailab/user/zhangziran/Dataset/Sim2Real_release"

RC.test = ED()
RC.test.rgb = "/ailab/user/zhangziran/Dataset/Sim2Real_release"
RC.test.evs = "/ailab/user/zhangziran/Dataset/Sim2Real_release"
