import os
from os.path import join, split, splitext
from tools import parse_path
import socket
from easydict import EasyDict as ED

mkdir = lambda x:os.makedirs(x, exist_ok=True)

# hostname = 'server'
# hostname = 'local' if 'dsw' not in socket.gethostname() else 'server'
source_path = ED()
source_path.train = ED()
source_path.test = ED()
# source_path.debug = ED()

source_path.train.source_path =  '/ailab/group/pjlab-sail/zhangziran/Syn_DATASET/v4_Gopro_lowlight_gray_no_noise/train_x8interpolated/'
source_path.test.source_path  = '/ailab/group/pjlab-sail/zhangziran/Syn_DATASET/v4_Gopro_lowlight_gray_no_noise/test_x8interpolated/'
source_path.train.target_path = '/ailab/user/zhangziran/Self_collected_DATASET/new_capture/train_x8interpolated/'
source_path.test.target_path  = '/ailab/user/zhangziran/Self_collected_DATASET/new_capture/test_x8interpolated/'


data_property = ED()
data_property.h = 720
data_property.w = 1280
data_property.fps = 240*8
data_property.mul = 8

source_path.train.target = ED()
source_path.train.target.base = source_path.train.target_path
source_path.train.target.events_path = os.path.join(source_path.train.target_path, 'EVS')
source_path.train.target.events_vis_path = os.path.join(source_path.train.target_path, 'EVSvis')
source_path.train.target.accint_path = os.path.join(source_path.train.target_path, 'AccINT')
source_path.train.target.event_vis_acc_path = os.path.join(source_path.train.target_path, f'EVSvis_acc{data_property.mul}')

source_path.test.target = ED()
source_path.test.target.base = source_path.test.target_path
source_path.test.target.events_path = os.path.join(source_path.test.target_path, 'EVS')
source_path.test.target.events_vis_path = os.path.join(source_path.test.target_path, 'EVSvis')
source_path.test.target.accint_path = os.path.join(source_path.test.target_path, 'AccINT')
source_path.test.target.event_vis_acc_path = os.path.join(source_path.test.target_path, f'EVSvis_acc{data_property.mul}')



from tools import parse_path
source_path.train.source_path_dict = parse_path(source_path.train.source_path, 2)
try:
    source_path.test.source_path_dict = parse_path(source_path.test.source_path, 2)
except FileNotFoundError:
    pass
# try:
#     source_path.debug.source_path_dict = parse_path(source_path.debug.source_path, 2)
# except FileNotFoundError:
#     pass


for p in list(source_path.train.target.keys()):
    try:
        mkdir(source_path.train.target[p])
    except:
        print(f"file does not exist: {source_path.train.target[p]}")

for p in list(source_path.test.target.keys()):
    try:
        mkdir(source_path.test.target[p])
    except:
        print(f"file does not exist: {source_path.test.target[p]}")


# for p in list(source_path.debug.target.keys()):
#     try:
#         mkdir(source_path.debug.target[p])
#     except:
#         print(f"file does not exist: {source_path.debug.target[p]}")