import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = models.densenet161()
  flops, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
  print('Flops:  ' + flops)
  print('Params: ' + params)