# generates moving dot(s)

# use it like this:
#v2e --synthetic_input=scripts.moving_dot --disable_slomo --dvs_aedat2=v2e.aedat --output_width=346 --output_height=260

# NOTE: There are nonintuitive effects of low contrast dot moving repeatedly over the same circle:
# The dot initially makes events and then appears to disappear. The cause is that the mean level of dot
# is encoded by the baseLogFrame which is initially at zero but increases to code the average of dot and background.
# Then the low contrast of dot causes only a single ON event on first cycle
import argparse

import numpy as np
import cv2
import os
from tqdm import tqdm
from v2ecore.base_synthetic_input import base_synthetic_input
from v2ecore.v2e_utils import *
import sys
from typing import Tuple, Optional
from .GOPRO_config import source_path, data_property, mkdir
logger = logging.getLogger(__name__)


class GOPRO_SyntheticInput_train(base_synthetic_input): # the class name should be the same as the filename, like in Java
    """ Generates moving dot
    """

    def __init__(self, *args, **kwargs) -> None:
        """ Constructs moving-dot class to make frames for v2e

        :param width: width of frames in pixels
        :param height: height in pixels
        :param avi_path: folder to write video to, or None if not needed
        :param preview: set true to show the pix array as cv frame
        """
        super().__init__(preview=kwargs['preview'])
        self.preview = kwargs['preview']
        self.w = data_property.h
        self.h = data_property.w
        self.fps = data_property.fps
        self.frame_number = 0
        self.log = sys.stdout
        self.key = kwargs['key']
        self.dict_path = source_path.train.source_path_dict
        # self.keys=list(self.dict_path.keys())
        self.file_list = self.dict_path[self.key]
        self.prev_key=''
        self.t = 0
        self.dt = 1/self.fps
        self.file_name = ''
        self.file_path = ''
        # parse saving path
        self.event_path = os.path.join(source_path.train.target.events_path, self.key)
        self.event_vis_path = os.path.join(source_path.train.target.events_vis_path, self.key)
        self.accint_path = os.path.join(source_path.train.target.accint_path, self.key)
        self.event_vis_acc_path = os.path.join(source_path.train.target.event_vis_acc_path, self.key)
        mkdir(self.event_path)
        mkdir(self.event_vis_acc_path)
        mkdir(self.event_vis_path)
        mkdir(self.event_vis_acc_path)

    def total_frames(self):
        """:returns: total number of frames"""
        frame_count=0
        frame_count += len(self.dict_path[self.key])
        return frame_count

    def next_frame(self) -> Tuple[Optional[np.ndarray], float]:
        """ Returns the next frame and its time, or None when finished

        :returns: (frame, time)
            If there are no more frames frame is None.
            time is in seconds.
        """
        try:
            self.file_path = self.file_list.pop(0)
        except:
            return (None, self.t)
        pix_arr = cv2.imread(self.file_path)
        pix_arr = cv2.cvtColor(pix_arr, cv2.COLOR_BGR2GRAY)
        self.file_name = os.path.splitext(os.path.split(self.file_path)[-1])[0]
        time=self.t
        self.t += self.dt
        return (pix_arr, time)


# if __name__ == "__main__":
#     m = GOPRO_train()
#     (fr, time) = m.next_frame()
#     with tqdm(total=m.total_frames(), desc='GOPRO_train', unit='fr') as pbar:  # instantiate progress bar
#         while fr is not None:
#             (fr, time) = m.next_frame()
#             pbar.update(1)
