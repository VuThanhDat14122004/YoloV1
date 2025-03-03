import xml.etree.ElementTree as ET
import os
import matplotlib.patches as patches
import torch
from torchvision import ops
import torch.nn.functional as F
import math

def parse_annotation(annotation_folder_path, img_file_path, jpg_folder):
    '''
    imgs_list_dir: list(N)
    gt_class_all: size = (N, M) with M is number of bounding box in each image,
                        M different in each image, N as above
    gt_boxes_all: size = (N, M, 4), (xmin, ymin, xmax, ymax)
    '''
    with open(img_file_path, 'r') as f:
        img_files = f.read().strip().split('\n')
    gt_class_all = []
    gt_boxes_all = []
    imgs_dir = []
    for img_file in img_files:
        jpg_file = img_file + '.jpg'
        xml_file = img_file + '.xml'
        xml_file_path = os.path.join(annotation_folder_path, xml_file)
        try:
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            all_object = root[6:]
        except:
            continue
        imgs_dir.append(os.path.join(jpg_folder,jpg_file))
        list_gt_class = []
        list_gt_boxes = []
        for object in all_object:
            list_gt_class.append(object[0].text)
            bndbox = object.find("bndbox")
            list_gt_boxes.append([float(bndbox[0].text), float(bndbox[1].text),
                                  float(bndbox[2].text), float(bndbox[3].text)])
        gt_boxes_all.append(torch.Tensor(list_gt_boxes))
        gt_class_all.append(list_gt_class)
    return imgs_dir, gt_class_all, gt_boxes_all

def intersection_over_union(predict, target):
    '''
    predict: tensor batch_size,S,S,4
    target: tensor batch_size,S,S,4
    '''
    result = torch.zeros((predict.shape[0], predict.shape[1], predict.shape[2]))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            for k in range(result.shape[2]):
                result[i][j][k]=IOU(predict[i][j][k], target[i][j][k])

    return result

def IOU(box1, box2):
    '''
    box: tensor, (4,) x_center, y_center, width, height so với cell i, j
    i: biểu thị ở trục y
    j: biểu thị ở trục x
    '''
    # box1
    x_topleft_1 = max(box1[0] - box1[2]/2,0)
    y_topleft_1 = max(box1[1] - box1[3]/2,0)
    x_botright_1 = min(box1[0] + box1[2]/2,1)
    y_botright_1 = min(box1[1] + box1[3]/2,1)
    area1 = (x_botright_1 - x_topleft_1)*(y_botright_1-y_topleft_1)
    # box2
    x_topleft_2 = max(box2[0] - box2[2]/2,0)
    y_topleft_2 = max(box2[1] - box2[3]/2,0)
    x_botright_2 = min(box2[0] + box2[2]/2,1)
    y_botright_2 = min(box2[1] + box2[3]/2,1)
    area2 = (x_botright_2 - x_topleft_2)*(y_botright_2-y_topleft_2)
    #cal iou
    x_u_min = max(x_topleft_1, x_topleft_2)
    y_u_min = max(y_topleft_1, y_topleft_2)
    x_u_max = min(x_botright_1, x_botright_2)
    y_u_max = min(y_botright_1, y_botright_2)


    
    union = (max(x_u_max-x_u_min,0))*(max(y_u_max - y_u_min,0))
    denominator = area2+area1-union

    return union/denominator

def encode_label(gt_class_all):
    '''
    gt_classes_all: 2-dimension
    '''
    dict_result = {}
    set_label = set()
    for i in gt_class_all:
        for j in i:
            set_label.add(j)
    list_label = list(set_label)
    for i in range(len(list_label)):
        dict_result.update({list_label[i]:i})
    return dict_result

def non_maxima_suppression(predictions, iou_threshold, prob_threshold, scale=64):
    '''
    predictions: 1x7x7x30
    '''
    bboxes = []
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            for k in range(predictions.shape[2]):
                class_of_box = torch.argmax(predictions[i][j][k][0:20]).item()
                if predictions[i][j][k][20] >= prob_threshold:
                    x_c_tall = k*scale + predictions[i][j][k][21]*scale
                    y_c_tall = j*scale + predictions[i][j][k][22]*scale
                    w_tall = predictions[i][j][k][23]*scale
                    h_tall = predictions[i][j][k][24]*scale
                    box_tall = [class_of_box, predictions[i][j][k][20], x_c_tall, y_c_tall, w_tall, h_tall]
                    bboxes.append(box_tall)
                if predictions[i][j][k][25] >= prob_threshold:
                    x_c_wide = k*scale + predictions[i][j][k][26]*scale
                    y_c_wide = j*scale + predictions[i][j][k][27]*scale
                    w_wide = predictions[i][j][k][28]*scale
                    h_wide = predictions[i][j][k][29]*scale
                    box_wide = [class_of_box, predictions[i][j][k][25], x_c_wide, y_c_wide, w_wide, h_wide]
                    bboxes.append(box_wide)
    box_entry = nms(bboxes, iou_threshold)
    return box_entry
def nms(bboxes, iou_threshold):
    '''
    bboxes: [[class_of_box, prob, x, y, w, h]]
    '''
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or IOU(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:])
            )
            < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms

# def decode_class(dict_class):
#     result = {v:k for k, v in dict_class.items()}
#     return result