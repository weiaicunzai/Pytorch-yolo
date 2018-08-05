"""YOLO loss function

Author: baiyu
"""

import torch
import torch.nn as nn

from conf import settings

class YOLOLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.S = settings.S
        self.B = settings.B
        self.l_coord = settings.LAMBDA_COORD
        self.l_noobj = settings.LAMBDA_NOOBJ
        self.img_size = settings.IMG_SIZE
        self.sum_squared = nn.MSELoss(size_average=False)

    def _comput_iou(self, pred_boxes, target_boxes):
        """compute iou of two set of boexs

        Args:
            pred_boxes:a 2D tensor(N, 4), N is the box number
            target_boxes:a 2D tensor(N, 4), N is the box number
        
        Returns:
            a tensor contains iou for every 2 corresponding boexs
        """

        pred_boxes *= self.img_size
        target_boxes *= self.img_size

        print(torch.equal(pred_boxes, target_boxes))

        #get top-left x y and bottom-right x y
        pred_corners = pred_boxes.clone()
        pred_corners[:, :2] = pred_boxes[:, :2] - 0.5 * pred_boxes[:, 2:]
        pred_corners[:, 2:] = pred_boxes[:, :2] + 0.5 * pred_boxes[:, 2:]

        target_corners = target_boxes.clone()
        target_corners[:, :2] = target_boxes[:, :2] - 0.5 * target_boxes[:, 2:]
        target_corners[:, 2:] = target_boxes[:, :2] + 0.5 * target_boxes[:, 2:]

        print(torch.equal(pred_corners, target_corners))
        #find the x, y coordinates of the intersection rectangle
        tl_x = torch.max(pred_corners[:, 0], target_corners[:, 0])
        tl_y = torch.max(pred_corners[:, 1], target_corners[:, 1])
        br_x = torch.min(pred_corners[:, 2], target_corners[:, 2])
        br_y = torch.min(pred_corners[:, 3], target_corners[:, 3])

        zeros = tl_x.clone().fill_(0)

        # + 1 means puls the boarder
        inter_area = torch.mul(torch.max(zeros, br_x - tl_x + 1), torch.max(zeros, br_y - tl_y + 1))
        pred_box_area = (pred_boxes[:, 2] + 1) * (pred_boxes[:, 3] + 1) # w * h
        target_box_area = (target_boxes[:, 2] + 1) * (target_boxes[:, 3] + 1)

        iou = inter_area / (pred_box_area + target_box_area - inter_area)

        return iou


    def forward(self, pred, target):

        #get target obj mask
        obj_mask = target[:, :, :, 4] > 0
        obj_mask = obj_mask.unsqueeze(-1).expand_as(target)

        #get no obj mask
        no_obj_mask = target[:, :, :, 4] == 0
        no_obj_mask = no_obj_mask.unsqueeze(-1).expand_as(pred)

        #pred no obj value according to target mask 
        #(batch_size * no_obj_num * 30)
        pred_no_obj = pred[no_obj_mask].view(-1, 30)

        #pred obj value according to target mask
        pred_obj = pred[obj_mask].view(-1, 30)

        #target no obj value (batch_size * obj_num)
        target_no_obj = target[no_obj_mask].view(-1, 30)

        #target obj value
        target_obj = target[obj_mask].view(-1, 30)

        #class_probs_loss

        #"""Note that the loss function only penalizes classification
        #error if an object is present in that grid cell"""
        class_preds = pred_obj[:, 10:]
        class_target = target_obj[:, 10:]
        class_probs_loss = self.sum_squared(class_preds, class_target)
        print(torch.equal(class_preds, class_target))

        #noobj_confidence_score_loss
        confidence_preds_no_obj = torch.cat((pred_no_obj[:, 4], pred_no_obj[:, 9]), dim=0)
        confidence_target_no_obj = torch.cat((target_no_obj[:, 4], target_no_obj[:, 9]), dim=0)
        no_obj_confidence_loss = self.sum_squared(confidence_preds_no_obj, confidence_target_no_obj)
        print(torch.equal(confidence_preds_no_obj, confidence_target_no_obj))

        #cell contains object confidence loss
        #loss function penalize confidence score both C1 and C2 for each cell

        #get bbox x, y, w, h 
        bbox_preds_obj = torch.cat((pred_obj[:, :4], pred_obj[:, 5:9]), dim=0)
        bbox_target_obj = torch.cat((target_obj[:, :4], target_obj[:, 5:9]), dim=0)

        print(torch.equal(bbox_preds_obj, bbox_preds_obj))

        iou = self._comput_iou(bbox_preds_obj, bbox_target_obj)
        return iou



yolo_loss = YOLOLoss()

from torch.autograd import Variable
t = Variable(torch.Tensor(2, 7, 7, 30))
p = Variable(torch.Tensor(2, 7, 7, 30))

#p[:, 2, 3, :].fill(1)

print(yolo_loss(t, t))


import numpy as np

a = np.arange(27).reshape(3, 3, 3)
#print(a[1, 2, :])

import cProfile

cProfile.runctx('yolo_loss(t, t)', globals(), locals())