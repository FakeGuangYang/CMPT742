from torch.utils.data import Dataset
import numpy as np
import os
import math
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import cv2


# generate default bounding boxes
def default_box_generator(layers, large_scale, small_scale):
    # input:
    # layers      -- a list of sizes of the output layers. in this assignment, it is set to [10,5,3,1].
    # large_scale -- a list of sizes for the larger bounding boxes. in this assignment, it is set to [0.2,0.4,0.6,0.8].
    # small_scale -- a list of sizes for the smaller bounding boxes. in this assignment, it is set to [0.1,0.3,0.5,0.7].

    # output:
    # boxes -- default bounding boxes, shape=[box_num,8]. box_num=4*(10*10+5*5+3*3+1*1) for this assignment.

    # TODO:
    # create an numpy array "boxes" to store default bounding boxes
    # you can create an array with shape [10*10+5*5+3*3+1*1,4,8], and later reshape it to [box_num,8]
    # the first dimension means number of cells, 10*10+5*5+3*3+1*1
    # the second dimension 4 means each cell has 4 default bounding boxes.
    # their sizes are [small_size,small_size], [large_size,large_size], [large_size*sqrt(2),large_size/sqrt(2)], [large_size/sqrt(2),large_size*sqrt(2)],
    # where small_size is the corresponding size in "small_scale" and large_size is the corresponding size in "large_scale".
    # for a cell in layer[i], you should use small_size=small_scale[i] and large_size=large_scale[i].
    # the last dimension 8 means each default bounding box has 8 attributes: [x_center, y_center, box_width, box_height, x_min, y_min, x_max, y_max]
    boxes = []
    for idx, grid_size in enumerate(layers):
        large_size = large_scale[idx]
        small_size = small_scale[idx]

        # calculate center coordinates and bounding boxes for each grid
        step = 1.0 / grid_size  # width/height of each grid
        for i in range(grid_size):
            for j in range(grid_size):
                x_center = (j + 0.5) * step
                y_center = (i + 0.5) * step

                # generate 4 default bounding boxes
                # 1. small scale square bounding box
                box1 = [x_center, y_center, small_size, small_size,
                        max(0, x_center - small_size / 2), max(0, y_center - small_size / 2),
                        min(1, x_center + small_size / 2), min(1, y_center + small_size / 2)]

                # 2. large scale square bounding box
                box2 = [x_center, y_center, large_size, large_size,
                        max(0, x_center - large_size / 2), max(0, y_center - large_size / 2),
                        min(1, x_center + large_size / 2), min(1, y_center + large_size / 2)]

                # 3. large scale rectangle bounding box (width is sqrt(2) * large_size and height is large_size / sqrt(2))
                box3 = [x_center, y_center, min(1, large_size * math.sqrt(2)), large_size / math.sqrt(2),
                        max(0, x_center - large_size * math.sqrt(2) / 2),
                        max(0, y_center - large_size / math.sqrt(2) / 2),
                        min(1, x_center + large_size * math.sqrt(2) / 2),
                        min(1, y_center + large_size / math.sqrt(2) / 2)]

                # 4. large scale rectangle bounding box (width is large_size / sqrt(2) and height is sqrt(2) * large_size)
                box4 = [x_center, y_center, large_size / math.sqrt(2), min(1, large_size * math.sqrt(2)),
                        max(0, x_center - large_size / math.sqrt(2) / 2),
                        max(0, y_center - large_size * math.sqrt(2) / 2),
                        min(1, x_center + large_size / math.sqrt(2) / 2),
                        min(1, y_center + large_size * math.sqrt(2) / 2)]

                # add all boxes to the list
                boxes.extend([box1, box2, box3, box4])
    # convert to numpy
    boxes = np.array(boxes)

    return boxes


# this is an example implementation of IOU.
# It is different from the one used in YOLO, please pay attention.
# you can define your own iou function if you are not used to the inputs of this one.
def iou(default_boxes, x_min, y_min, x_max, y_max):
    # input:
    # default_boxes -- [num_of_boxes, 8], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_center, y1_center, width, height, x1_min, y1_min, x1_max, y1_max].
    # x_min, y_min, x_max, y_max -- another box (box_r)

    # output:
    # ious between the "default_boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]

    inter = np.maximum(np.minimum(default_boxes[:, 6], x_max) - np.maximum(default_boxes[:, 4], x_min), 0) * np.maximum(
        np.minimum(default_boxes[:, 7], y_max) - np.maximum(default_boxes[:, 5], y_min), 0)
    area_a = (default_boxes[:, 6] - default_boxes[:, 4]) * (default_boxes[:, 7] - default_boxes[:, 5])
    area_b = (x_max - x_min) * (y_max - y_min)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def iou_nms(boxes, x_min, y_min, x_max, y_max):
    # input:
    # boxes -- [num_of_boxes, 4], a list of boxes stored as [box_1,box_2, ...], where box_1 = [x1_min, y1_min, x1_max, y1_max].
    # x_min, y_min, x_max, y_max -- another box (box_r)

    # output:
    # ious between the "default_boxes" and the "another box": [iou(box_1,box_r), iou(box_2,box_r), ...], shape = [num_of_boxes]

    inter = np.maximum(np.minimum(boxes[:, 2], x_max) - np.maximum(boxes[:, 0], x_min), 0) * np.maximum(
        np.minimum(boxes[:, 3], y_max) - np.maximum(boxes[:, 1], y_min), 0)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (x_max - x_min) * (y_max - y_min)
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-8)


def match(ann_box, ann_confidence, default_boxes, threshold, cat_id, x_min, y_min, x_max, y_max):
    # input:
    # ann_box                 -- [num_of_boxes,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [num_of_boxes,number_of_classes], ground truth class labels to be updated
    # boxs_default            -- [num_of_boxes,8], default bounding boxes
    # threshold               -- if a default bounding box and the ground truth bounding box have iou>threshold, then this default bounding box will be used as an anchor
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box

    # compute iou between the default bounding boxes and the ground truth bounding box
    cat_id = int(cat_id)
    ious = iou(default_boxes, x_min, y_min, x_max, y_max)

    ious_true = ious > threshold
    # TODO:
    # update ann_box and ann_confidence, with respect to the ious and the default bounding boxes.
    # if a default bounding box and the ground truth bounding box have iou>threshold, then we will say this default bounding box is carrying an object.
    # this default bounding box will be used to update the corresponding entry in ann_box and ann_confidence
    for idx in range(len(ious)):
        if ious_true[idx]:
            # calculate sizes for ann_box
            px, py, pw, ph = default_boxes[idx, :4]
            tx = ((x_max + x_min) / 2 - px) / pw
            ty = ((y_max + y_min) / 2 - py) / ph
            tw = np.log((x_max - x_min) / pw + 1e-8)
            th = np.log((y_max - y_min) / ph + 1e-8)

            # update ann_box and ann_confidence
            ann_box[idx] = [tx, ty, tw, th]
            ann_confidence[idx][cat_id] = 1
            ann_confidence[idx][-1] = 0

    # if there's no iou > threshold, we have to assign at least one bounding box
    best_match_idx = np.argmax(ious)
    # TODO:
    # make sure at least one default bounding box is used
    # update ann_box and ann_confidence (do the same thing as above)
    # calculate best match ann_box
    px, py, pw, ph = default_boxes[best_match_idx, :4]
    tx = ((x_max + x_min) / 2 - px) / pw
    ty = ((y_max + y_min) / 2 - py) / ph
    tw = np.log((x_max - x_min) / pw + 1e-8)
    th = np.log((y_max - y_min) / ph + 1e-8)

    # update ann_box and ann_confidence for the best match
    ann_box[best_match_idx] = [tx, ty, tw, th]
    ann_confidence[best_match_idx, cat_id] = 1
    ann_confidence[best_match_idx, -1] = 0


class COCO(Dataset):
    def __init__(self, imgdir, anndir, class_num, default_boxes, train=True, image_size=320):
        self.main_dir = os.getcwd()
        self.train = train
        self.imgdir = self.main_dir + imgdir
        self.anndir = self.main_dir + anndir
        self.class_num = class_num

        # overlap threshold for deciding whether a bounding box carries an object or no
        self.threshold = 0.5
        self.default_boxes = default_boxes
        self.box_num = len(self.default_boxes)

        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size

        # notice:
        # you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train
        train_size = int(len(self.img_names) * 0.9)
        self.img_names = self.img_names[:train_size] if self.train else self.img_names[train_size:]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        ann_box = np.zeros([self.box_num, 4], np.float32)  # bounding boxes
        ann_confidence = np.zeros([self.box_num, self.class_num], np.float32)  # one-hot vectors
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background

        ann_confidence[:, -1] = 1  # the default class for all cells is set to "background"

        img_name = self.imgdir + self.img_names[index]
        ann_name = self.anndir + self.img_names[index][:-3] + "txt"
        # TODO:
        # 1. prepare the image [3,320,320], by reading image "img_name" first.
        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        # 3. use the above function "match" to update ann_box and ann_confidence, for each bounding box in "ann_name".
        # 4. Data augmentation. You need to implement random cropping first. You can try adding other augmentations to get better results.
        # get main directory to find those files

        image = cv2.imread(img_name)
        bounding_boxes = []
        h, w, _ = image.shape

        # find bounding box for each line
        with open(ann_name, 'r') as f:
            for line in f.readlines():
                data = line.strip('\n').split(' ')
                # data = [cat_id, x_min, y_min, width, height]
                cat_id = int(data[0])
                x_min, y_min, width, height = map(float, data[1:])
                # calculate x_max and y_max
                x_max = x_min + width
                y_max = y_min + height
                # normalize to [0, 1]
                bounding_boxes.append([x_min / w, y_min / h, x_max / w, y_max / h, cat_id])

        # data augmentation
        train_transforms = A.Compose([
            A.BBoxSafeRandomCrop(erosion_rate=0.2, p=1.0),
            A.HorizontalFlip(),
            A.Rotate(limit=10, p=1),
            A.RandomBrightnessContrast(p=0.3),
            A.Resize(self.image_size, self.image_size),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels'])
        )
        transformed = train_transforms(
            image=image,
            bboxes=[(x_min, y_min, x_max, y_max) for x_min, y_min, x_max, y_max, _ in bounding_boxes],
            labels=[cat_id for _, _, _, _, cat_id in bounding_boxes]
        )
        image, augmented_bounding_boxes, augmented_cat_id = transformed['image'], transformed['bboxes'], transformed['labels']

        # to use function "match":
        # match(ann_box,ann_confidence,self.default_boxes,self.threshold,class_id,x_min,y_min,x_max,y_max)
        # where [x_min,y_min,x_max,y_max] is from the ground truth bounding box, normalized with respect to the width or height of the image.
        for cat_id, [x_min, y_min, x_max, y_max] in zip(augmented_cat_id, augmented_bounding_boxes):
            match(ann_box, ann_confidence, self.default_boxes, self.threshold, cat_id, x_min, y_min, x_max, y_max)

        # note: please make sure x_min,y_min,x_max,y_max are normalized with respect to the width or height of the image.
        # For example, point (x=100, y=200) in an image with (width=1000, height=500) will be normalized to (x/width=0.1,y/height=0.4)

        return image, ann_box, ann_confidence
