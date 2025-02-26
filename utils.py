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
    x_topleft_1 = box1[0] - box1[2]/2
    y_topleft_1 = box1[1] - box1[3]/2
    x_botright_1 = box1[0] + box1[2]/2
    y_botright_1 = box1[1] + box1[3]/2
    area1 = (x_botright_1 - x_topleft_1)*(y_botright_1-y_topleft_1)
    # box2
    x_topleft_2 = box2[0] - box2[2]/2
    y_topleft_2 = box2[1] - box2[3]/2
    x_botright_2 = box2[0] + box2[2]/2
    y_botright_2 = box2[1] + box2[3]/2
    area2 = (x_botright_2 - x_botright_1)*(y_botright_2-y_botright_1)
    #cal iou
    x_u_min = max(x_topleft_1, x_topleft_2)
    y_u_min = max(y_topleft_1, y_topleft_2)
    x_u_max = min(x_botright_1, x_botright_2)
    y_u_max = min(y_botright_1,y_botright_2)

    union = (x_u_max-x_u_min)*(y_u_max - y_u_min)
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

# def decode_class(dict_class):
#     result = {v:k for k, v in dict_class.items()}
#     return result