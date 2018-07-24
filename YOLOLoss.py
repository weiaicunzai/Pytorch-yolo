"""YOLO loss function

Author: baiyu
"""

import torch
import torch.nn as nn

from conf import settings

class YOLOLoss(nn.Module):

    def __init__(self):
        self.S = settings.S
        self.B = settings.B
        self.l_coord = settings.LAMBDA_COORD
        self.l_noobj = settings.LAMBDA_NOOBJ



yolo_loss = YOLOLoss()

