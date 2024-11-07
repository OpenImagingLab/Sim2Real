import os
from natsort import natsorted

def parse_path(folder, fdepth):
    file_dict = {}
    if fdepth == 1:
        file_dict.update({
            os.path.split(folder)[-1]:[os.path.join(folder, file) for file in natsorted(os.listdir(folder))]
        })
    else:
        for subfolder in os.listdir(folder):
            sfolder = os.path.join(folder, subfolder)
            file_dict.update({
                subfolder:[os.path.join(sfolder, file) for file in natsorted(os.listdir(sfolder))]
            })
    return file_dict