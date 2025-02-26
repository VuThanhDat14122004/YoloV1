import torch
import os
import pandas as pd
from PIL import Image
from utils import parse_annotation, encode_label
import cv2

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, Annotations_path, img_path, jpg_path, S=7, B=2, C=20, img_size=(448,448)):
        self.Annotations_path = Annotations_path
        self.img_path = img_path
        self.jpg_path = jpg_path
        self.S = S
        self.B = B
        self.C = C
        self.img_size = img_size
        self.scale_resize = []
        self.img_data, self.gt_classes_all,\
            self.gt_boxes_all, self.name2idx = self.get_data()
        

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, index):
        '''
        return image, label_matrix
        image: Image
        label_matrix: 
        [:,:,(0 or 1), x_center, y_center, width, height]
                         so với S
        torch.size(S*S*25), 0 -> 20):class, 20: 1, 21-> 25): coords
        '''
        gt_classes = self.gt_classes_all[index]
        gt_boxes = self.gt_boxes_all[index]
        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        scale_resize = self.scale_resize[index]
        for idxx in range(len(gt_classes)):
            class_label = gt_classes[idxx]
            class_label = self.name2idx[class_label]
            x_min, y_min, x_max, y_max = gt_boxes[idxx]
            x_min, x_max = x_min*scale_resize[0], x_max*scale_resize[0]
            y_min, y_max = y_min*scale_resize[1], y_max*scale_resize[1]
            width = x_max - x_min
            height = y_max - y_min
            x_center = x_min + width/2
            y_center = y_min + height/2
            i,j = int(self.S*(y_center/self.img_size[0])),\
                int(self.S*(x_center/self.img_size[1]))
            x_cell, y_cell = self.S*(x_center/self.img_size[1])-j,\
                self.S*(y_center/self.img_size[0])-i
            width_cell, height_cell = (
                (width/self.img_size[1])*self.S,
                (height/self.img_size[0])*self.S
            )
            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coords = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coords
                label_matrix[i,j,class_label]=1
        return self.img_data[index], label_matrix
    def get_data(self):
        img_data = []
        img_paths, gt_classes_all,\
            gt_boxes_all = parse_annotation(self.Annotations_path,
                                            self.img_path,
                                            self.jpg_path)
        for i  in range(len(img_paths)):
            scale_x, scale_y = 1,1
            image = cv2.imread(img_paths[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            scale_x = self.img_size[1]/image.shape[1] # image.shape (height, width)
            scale_y = self.img_size[0]/image.shape[0]
            image = cv2.resize(image,self.img_size) # (img_size, img_size, 3)
            image = torch.permute(torch.tensor(image).float(), dims=(2,0,1))
            img_data.append(image)
            self.scale_resize.append([scale_x, scale_y])
        encode = encode_label(gt_classes_all)
        return img_data, gt_classes_all, gt_boxes_all, encode
