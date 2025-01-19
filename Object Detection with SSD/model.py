import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


def SSD_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    # input:
    # pred_confidence -- the predicted class labels from SSD, [batch_size, num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [batch_size, num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, num_of_boxes, 4]
    #
    # output:
    # loss -- a single number for the value of the loss function, [1]

    # TODO: write a loss function for SSD
    # For confidence (class labels), use cross entropy (F.cross_entropy)
    # You can try F.binary_cross_entropy and see which loss is better
    # For box (bounding boxes), use smooth L1 (F.smooth_l1_loss)

    # Note that you need to consider cells carrying objects and empty cells separately.
    # I suggest you to reshape confidence to [batch_size*num_of_boxes, num_of_classes]
    # and reshape box to [batch_size*num_of_boxes, 4].
    # Then you need to figure out how you can get the indices of all cells carrying objects,
    # and use confidence[indices], box[indices] to select those cells.
    # Flatten the dimensions for easier indexing
    batch_size, num_boxes, num_classes = pred_confidence.shape
    pred_confidence = pred_confidence.view(-1, num_classes)
    ann_confidence = ann_confidence.view(-1, num_classes)
    pred_box = pred_box.view(-1, 4)
    ann_box = ann_box.view(-1, 4)

    # Identify positive (contains object) and negative (no object) cells
    obj_indices = torch.argmax(ann_confidence, dim=1) != 3
    no_obj_indices = ~obj_indices
    # print(obj_indices.shape)
    # print(pred_confidence.shape)

    # Confidence loss
    # Positive (object) cells
    cls_loss_obj = F.binary_cross_entropy(pred_confidence[obj_indices], ann_confidence[obj_indices])

    # Negative (no object) cells - multiplied by 3 to balance the effect
    cls_loss_no_obj = F.binary_cross_entropy(pred_confidence[no_obj_indices], ann_confidence[no_obj_indices])

    confidence_loss = cls_loss_obj + 3 * cls_loss_no_obj

    # Localization (bounding box) loss for positive cells
    box_loss = F.smooth_l1_loss(pred_box[obj_indices], ann_box[obj_indices], reduction='mean')

    # Total loss
    loss = confidence_loss + box_loss
    return loss


class SSD(nn.Module):

    def __init__(self, class_num):
        super(SSD, self).__init__()

        self.class_num = class_num  # num_of_classes, in this assignment, 4: cat, dog, person, background

        # TODO: define layers
        # Define layers as class attributes
        self.layers = nn.ModuleList()
        for layer in [
            (3, 64, 3, 2, 1),
            (64, 64, 3, 1, 1),
            (64, 64, 3, 1, 1),
            (64, 128, 3, 2, 1),
            (128, 128, 3, 1, 1),
            (128, 128, 3, 1, 1),
            (128, 256, 3, 2, 1),
            (256, 256, 3, 1, 1),
            (256, 256, 3, 1, 1),
            (256, 512, 3, 2, 1),
            (512, 512, 3, 1, 1),
            (512, 512, 3, 1, 1),
            (512, 256, 3, 2, 1),
            'conv10x10',
            (256, 256, 1, 1, 0),
            (256, 256, 3, 2, 1),
            'conv5x5',
            (256, 256, 1, 1, 0),
            (256, 256, 3, 1, 0),
            'conv3x3',
            (256, 256, 1, 1, 0),
            [256, 256, 3, 1, 0],  # no batch normalization
            'conv1x1'
        ]:
            if isinstance(layer, tuple):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(layer[0], layer[1], kernel_size=layer[2], stride=layer[3], padding=layer[4]),
                        nn.BatchNorm2d(layer[1]),
                        nn.ReLU()
                    )
                )
            elif isinstance(layer, list):
                self.layers.append(
                    nn.Sequential(
                        nn.Conv2d(layer[0], layer[1], kernel_size=layer[2], stride=layer[3], padding=layer[4]),
                        nn.ReLU()
                    )
                )
            elif layer == 'conv1x1':
                self.layers.append(nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0))
                self.layers.append(nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0))
            elif isinstance(layer, str):
                # Placeholder for custom layers (10x10, 5x5, 3x3, 1x1 layers)
                # Define separate layers for bounding box and confidence predictions
                self.layers.append(nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1))
                self.layers.append(nn.Conv2d(256, 16, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]

        # x = x/255.0 #normalize image. If you already normalized your input image in the dataloader, remove this line.

        # TODO: define forward
        x = x.float()
        confidences = {}
        bounding_boxes = {}

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Sequential):  # Standard conv layer with batch normalization and ReLU
                x = layer(x)
            else:  # Custom conv layers for bounding boxes and confidences
                if i % 2 == 0:  # bounding box conv
                    bounding_box = layer(x)
                    bounding_boxes[f"layer_{i}"] = torch.flatten(bounding_box, start_dim=2)
                else:  # confidence conv
                    confidence = layer(x)
                    confidences[f"layer_{i}"] = torch.flatten(confidence, start_dim=2)

        # Concatenate bounding boxes and confidences along last dimension
        bounding_boxes = torch.cat(tuple(bounding_boxes.values()), 2)
        confidences = torch.cat(tuple(confidences.values()), 2)

        # Reshape and apply softmax to confidence scores
        bboxes = bounding_boxes.permute(0, 2, 1).reshape(-1, 540, self.class_num)
        confidence = F.softmax(confidences.permute(0, 2, 1).reshape(-1, 540, self.class_num), dim=-1)

        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,4*(10*10+5*5+3*3+1*1),num_of_classes]
        # bboxes - [batch_size,4*(10*10+5*5+3*3+1*1),4]
        # print("Bounding boxes output shape:", bboxes.shape)
        # print("Confidence scores output shape:", confidence.shape)

        return confidence, bboxes
