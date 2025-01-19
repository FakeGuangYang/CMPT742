import numpy as np
import cv2
from dataset import iou, iou_nms

# use [blue, green, red] to represent different classes
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def visualize_pred(windowname, pred_confidence, pred_box, ann_confidence, ann_box, image_, boxs_default):
    # input:
    # windowname      -- the name of the window to display the images
    # pred_confidence -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # pred_box        -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # ann_confidence  -- the ground truth class labels, [num_of_boxes, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [num_of_boxes, 4]
    # image_          -- the input image to the network
    # boxs_default    -- default bounding boxes, [num_of_boxes, 8]

    _, class_num = pred_confidence.shape
    # class_num = 4
    class_num = class_num - 1
    # class_num = 3 now, because we do not need the last class (background)

    # image1: draw ground truth bounding boxes on image1
    # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)
    # image3: draw network-predicted bounding boxes on image3
    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
    image = np.transpose(image_, (1, 2, 0)).astype(np.uint8)
    image1 = np.zeros(image.shape, np.uint8)
    image2 = np.zeros(image.shape, np.uint8)
    image3 = np.zeros(image.shape, np.uint8)
    image4 = np.zeros(image.shape, np.uint8)
    image1[:] = image[:]
    image2[:] = image[:]
    image3[:] = image[:]
    image4[:] = image[:]

    # draw ground truth
    height, width, _ = image.shape
    for i in range(len(ann_confidence)):
        for j in range(class_num):
            if ann_confidence[i, j] > 0.5:  # if the network/ground_truth has high confidence on cell[i] with class[j]
                # TODO:
                # image1: draw ground truth bounding boxes on image1
                # image2: draw ground truth "default" boxes on image2 (to show that you have assigned the object to the correct cell/cells)

                # you can use cv2.rectangle as follows:
                # start_point = (x1, y1) #top left corner, x1<x2, y1<y2
                # end_point = (x2, y2) #bottom right corner
                # color = colors[j] #use red green blue to represent different classes
                # thickness = 2
                # cv2.rectangle(image?, start_point, end_point, color, thickness)
                # Ground truth bounding box
                tx, ty, tw, th = ann_box[i]
                px, py, pw, ph = boxs_default[i, :4]
                # calculate ground truth bouding box center and width and height - [gx, gy, gw, gh]
                gx = pw * tx + px
                gy = ph * ty + py
                gw = pw * np.exp(tw)
                gh = ph * np.exp(th)
                # calculate for visualization ground truth bounding boxes
                x_min = max(0, int((gx - gw / 2) * width))
                y_min = max(0, int((gy - gh / 2) * height))
                x_max = min(width, int((gx + gw / 2) * width))
                y_max = min(height, int((gy + gh / 2) * height))
                # visualization
                cv2.rectangle(image1, (x_min, y_min), (x_max, y_max), colors[j], 2)

                # Ground truth "default" box
                def_x_min = int(boxs_default[i, 4] * width)
                def_y_min = int(boxs_default[i, 5] * height)
                def_x_max = int(boxs_default[i, 6] * width)
                def_y_max = int(boxs_default[i, 7] * height)
                # visualization
                cv2.rectangle(image2, (def_x_min, def_y_min), (def_x_max, def_y_max), colors[j], 2)

        # pred
        for i in range(len(pred_confidence)):
            for j in range(class_num):
                if pred_confidence[i, j] > 0.5:
                    # TODO:
                    # image3: draw network-predicted bounding boxes on image3
                    # image4: draw network-predicted "default" boxes on image4 (to show which cell does your network think that contains an object)
                    # Predicted bounding box
                    tx, ty, tw, th = pred_box[i]
                    px, py, pw, ph = boxs_default[i, :4]
                    # calculate prediction bouding box center and width and height - [gx, gy, gw, gh]
                    gx = pw * tx + px
                    gy = ph * ty + py
                    gw = pw * np.exp(tw)
                    gh = ph * np.exp(th)
                    # calculate for visualization ground truth bounding boxes
                    x_min = max(0, int((gx - gw / 2) * width))
                    y_min = max(0, int((gy - gh / 2) * height))
                    x_max = min(width, int((gx + gw / 2) * width))
                    y_max = min(height, int((gy + gh / 2) * height))
                    # visualization
                    cv2.rectangle(image3, (x_min, y_min), (x_max, y_max), colors[j], 2)
                    # add class and confidence
                    cls = {'0': 'cat', '1': 'dog', '2': 'person'}
                    cv2.putText(image3, f"{cls[str(j)]}" + ":" + str(round(pred_confidence[i, j], 2)), (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[j], 2)

                    # Predicted "default" box
                    def_x_min = int(boxs_default[i, 4] * width)
                    def_y_min = int(boxs_default[i, 5] * height)
                    def_x_max = int(boxs_default[i, 6] * width)
                    def_y_max = int(boxs_default[i, 7] * height)
                    # visualization
                    cv2.rectangle(image4, (def_x_min, def_y_min), (def_x_max, def_y_max), colors[j], 2)

    # combine four images into one
    h, w, _ = image1.shape
    image = np.zeros([h * 2, w * 2, 3], np.uint8)
    image[:h, :w] = image1
    image[:h, w:] = image2
    image[h:, :w] = image3
    image[h:, w:] = image4
    cv2.imshow(windowname + " [[gt_box,gt_dft],[pd_box,pd_dft]]", image)
    cv2.waitKey(1)
    # if you are using a server, you may not be able to display the image.
    # in that case, please save the image using cv2.imwrite and check the saved image for visualization.


def non_maximum_suppression(confidence_, box_, overlap=0.5, threshold=0.5):
    # TODO: non maximum suppression
    # input:
    # confidence_  -- the predicted class labels from SSD, [num_of_boxes, num_of_classes]
    # box_         -- the predicted bounding boxes from SSD, [num_of_boxes, 4]
    # boxs_default -- default bounding boxes, [num_of_boxes, 8]
    # overlap      -- if two bounding boxes in the same class have iou > overlap, then one of the boxes must be suppressed
    # threshold    -- if one class in one cell has confidence > threshold, then consider this cell carrying a bounding box with this class.

    # output:
    # depends on your implementation.
    # if you wish to reuse the visualize_pred function above, you need to return a "suppressed" version of confidence [5,5, num_of_classes].
    # you can also directly return the final bounding boxes and classes, and write a new visualization function for that.
    num_classes = confidence_.shape[1] - 1  # Exclude background class
    suppressed_confidence = np.zeros_like(confidence_)
    suppressed_boxes = np.zeros_like(box_)

    for cls in range(num_classes):  # Iterate over all classes except background
        class_confidence = confidence_[:, cls]
        indices = np.where(class_confidence > threshold)[0]  # Filter by confidence threshold

        # Sort indices by confidence scores in descending order
        sorted_indices = indices[np.argsort(-class_confidence[indices])]
        selected_indices = []

        while len(sorted_indices) > 0:
            # Select the index with the highest confidence
            current = sorted_indices[0]
            selected_indices.append(current)
            sorted_indices = sorted_indices[1:]  # Remove the current index

            # Compute IOU with the remaining boxes
            ious = iou_nms(box_[sorted_indices], *box_[current, :4])
            sorted_indices = sorted_indices[ious <= overlap]  # Suppress boxes with IOU > overlap

        # Update the output arrays with the selected indices
        suppressed_confidence[selected_indices, cls] = confidence_[selected_indices, cls]
        suppressed_boxes[selected_indices] = box_[selected_indices]

    return suppressed_confidence, suppressed_boxes


def generate_mAP(pred_confidence, pred_boxes, true_confidence, true_boxes, iou_threshold=0.5):
    # TODO: Generate mAP
    num_classes = pred_confidence.shape[1] - 1  # Exclude background class
    average_precisions = []

    for cls in range(num_classes):  # Iterate over all classes except background
        # Get predicted and true boxes for the current class
        pred_class_indices = np.where(pred_confidence[:, cls] > 0.5)[0]
        pred_class_boxes = pred_boxes[pred_class_indices]
        pred_class_scores = pred_confidence[pred_class_indices, cls]

        true_class_indices = np.where(true_confidence[:, cls] > 0.5)[0]
        true_class_boxes = true_boxes[true_class_indices]

        # Sort predicted boxes by confidence scores in descending order
        sorted_indices = np.argsort(-pred_class_scores)
        pred_class_boxes = pred_class_boxes[sorted_indices]

        # Match predictions to ground truth
        tp = np.zeros(len(pred_class_boxes))
        fp = np.zeros(len(pred_class_boxes))
        matched_gt_indices = set()

        for i, pred_box in enumerate(pred_class_boxes):
            # Compute IOU with all ground truth boxes for this class
            ious = iou(true_class_boxes, *pred_box)
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            if best_iou >= iou_threshold and best_iou_idx not in matched_gt_indices:
                tp[i] = 1  # True Positive
                matched_gt_indices.add(best_iou_idx)
            else:
                fp[i] = 1  # False Positive

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
        recall = tp_cumsum / len(true_class_boxes)

        # Compute Average Precision (AP) using 11-point interpolation
        recall_levels = np.linspace(0, 1, 11)
        precisions_at_recall_levels = []
        for recall_level in recall_levels:
            precisions = precision[recall >= recall_level] if np.any(recall >= recall_level) else [0]
            precisions_at_recall_levels.append(max(precisions))
        ap = np.mean(precisions_at_recall_levels)
        average_precisions.append(ap)

    # Return mean of all APs as mAP
    return np.mean(average_precisions)
