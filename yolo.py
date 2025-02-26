import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import *
from model import *
from utils import *
from loss import *
torch.autograd.set_detect_anomaly(True)


seed = 113
torch.manual_seed(seed)

learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
weight_decay=0
epochs=100
num_workers=2
pin_memory=True
load_model=False
load_model_file="yolotrain.pth"
annotation_dir = "VOC2007/Annotations"
train_dir = "VOC2007/ImageSets/Layout/trainval.txt"
val_dir = "VOC2007/ImageSets/Layout/val.txt"
jpg_dir = "VOC2007/JPEGImages"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, bboxes):
        for tf in self.transforms:
            img, bboxes = tf(img), bboxes
        
        return img, bboxes

transform = Compose([transforms.Resize((448,448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_function):
    pbar = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x,y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_function(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss = loss.item())

    print(f"mean loss {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    loss_fn = Yolov1Loss()
    # if load_model:
    #     load_checkpoint(torch.load(load_model_file), model, optimizer)
    train_dataset = VOCDataset(annotation_dir, train_dir, jpg_dir)
    val_dataset = VOCDataset(annotation_dir, val_dir, jpg_dir)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        drop_last=True,
    )

    for epoch in range(epochs):
        # compute mAP

        #

        # compute mean_avg_precision

        #


        train_fn(train_loader, model, optimizer, loss_fn)

if __name__ == "__main__":
    main()