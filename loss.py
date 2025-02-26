import torch
import torch.nn as nn
from utils import intersection_over_union

class Yolov1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(Yolov1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5) # bacthx7x7x30
        iou_b1 = intersection_over_union(predictions[:,:,:,21:25], target[:,:,:,21:25])
        iou_b2 = intersection_over_union(predictions[:,:,:,26:30], target[:,:,:,21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0) # best_box (argmax)is wide or tall box 0 or 1
        exists_box = target[:,:,:,20].unsqueeze(3) # identity_obj_i = 1 if object exists in cell i, else 0, batchx7x7x1
        # Box coordinates
        # best_box(batchx7x7) -> (batchx7x7x4) with best_box[:,:,:] = [0,0,0,0] or [1,1,1,1]
        best_box = best_box.unsqueeze(3).to(predictions.device)
        box_predictions = exists_box * (
            (best_box * predictions[:,:,:,26:30] + (1-best_box) * predictions[:,:,:,21:25])
            # (1 or 0)*wide box                      (1 or 0) * tall box
        )
        # avoid inline operation in gradient object
        box_predictions_new = box_predictions.clone()
        box_predictions_new[:,:,:, 2:4] = torch.sign(box_predictions[:,:,:, 2:4]) * torch.sqrt(
            torch.abs(box_predictions[:,:,:, 2:4]) + 1e-6
        )
        box_predictions = box_predictions_new

        box_targets = exists_box * target[:,:,:,21:25]
        # avoid inline operation in gradient object
        box_targets_new = box_targets.clone()
        box_targets_new[:,:,:, 2:4] = torch.sqrt(box_targets[:,:,:, 2:4])# sqrt(w,h)
        box_targets = box_targets_new
        
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )
        # object loss
        pred_box = (
            best_box*predictions[:,:,:,25:26] + (1-best_box) * predictions[:,:,:,20:21]
        )
        # (N,S,S,1) -> (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box*pred_box),
            torch.flatten(exists_box*target[:,:,:,20:21])
        )
        # No object loss
        # (N, S, S, 1) -> (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box)*predictions[:,:,:,20:21], start_dim=1),
            torch.flatten((1-exists_box)*target[:,:,:,20:21], start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten((1-exists_box)*predictions[:,:,:,25:26], start_dim=1),
            torch.flatten((1-exists_box)*target[:,:,:,20:21], start_dim=1)
        )
        # class loss
        class_loss = self.mse(
            torch.flatten(exists_box*predictions[:,:,:,:20], end_dim=-2),
            torch.flatten(exists_box*target[:,:,:,:20], end_dim=-2)
        )

        loss = (
            self.lambda_coord*box_loss
            + object_loss
            + self.lambda_noobj* no_object_loss
            + class_loss
        )

        return loss