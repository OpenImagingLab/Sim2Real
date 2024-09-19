from easydict import EasyDict as ED
from copy import deepcopy as dcopy

model_arch_config = ED()
model_arch_config.TimeLens = ED()
model_arch_config.TimeLens.train_config_dataloader = 'loader_timelens_mix'
model_arch_config.TimeLens.val_config_dataloader = 'loader_timelens_mix'
model_arch_config.TimeLens.num_bins = 5
