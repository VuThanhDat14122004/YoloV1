import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import *
from model import *
from utils import *
from loss import *
torch.autograd.set_detect_anomaly(True)


seed = 113
torch.manual_seed(seed)

learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 64
weight_decay=1e-5
epochs=100
num_workers=2
pin_memory=True
load_model=False
load_model_file="yolotrain.pth"
annotation_dir = "VOC2007/Annotations"
train_dir = "VOC2007/ImageSets/Layout/trainval.txt"
val_dir = "VOC2007/ImageSets/Layout/val.txt"
jpg_dir = "VOC2007/JPEGImages"
all_positive = 0


model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(device)
optimizer = optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay
)
loss_fn = Yolov1Loss()
loss_eval = Yolov1Loss()
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
            shuffle=False,
            drop_last=True,
        )

encode_label = train_dataset.name2idx
decode_label = {v:k for k,v in encode_label.items()}

# calculate FN + TP for mAP
for classes in train_dataset.gt_classes_all:
    all_positive += len(classes)
#

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
        loss = loss_function(out,y) # sẽ tính tổng số FP và tổng số TP ở đây
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss = loss.item())
    print(f"mean loss {sum(mean_loss)/len(mean_loss)}")
    return sum(mean_loss)/len(mean_loss)

def eval_fn(val_loader, model, loss_function):
    pbar = tqdm(val_loader, leave=True)
    mean_loss = []
    for batch_idx, (x,y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_function(out,y)
        mean_loss.append(loss.item())
        pbar.set_postfix(loss = loss.item())
    print(f"mean loss {sum(mean_loss)/len(mean_loss)}")
    return sum(mean_loss)/len(mean_loss)

    
def save_checkpoint(model, optimizer, filename, epoch, avg_loss):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "loss": avg_loss
    }
    torch.save(checkpoint, filename)
    print("Checkpoint saved")

def load_checkpoint(file_name, model, optimizer):
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded, epoch: {epoch}, loss: {loss}")
    return model, optimizer, epoch, loss

def main(mode, check_point_path=None):
    if mode == "train":
        model.eval()
        for epoch in range(epochs):
            # pred sample box by non maxima suppression
            img_sample, label_sample = train_dataset.__getitem__(0)
            # img_shape: 3x448x448
            pred_sample = model(img_sample.unsqueeze(0).to(device))
            # bboxes: [[class_of_box, prob, x, y, w, h]]
            pred_sample = pred_sample.view(1, 7, 7, 30)
            bboxes = non_maxima_suppression(pred_sample, 0.3, 0.5)
            fig, axes = plt.subplots(1,1)
            #imshow
            axes.imshow(img_sample.permute(1,2,0).long())
            if len(bboxes) != 0:
                for box in bboxes:
                    box = [int(i) for i in box]
                    x,y,w,h = box[2], box[3], box[4], box[5]
                    rect = plt.Rectangle((x-w/2,y-h/2),w,h,linewidth=1,edgecolor='r',facecolor='none')
                    axes.add_patch(rect)
                    axes.text(x-w/2, y-h/2, s=decode_label[box[0]], color='r')
            
            plt.show()
            # compute mean_avg_precision

            #
            avg_loss = train_fn(train_loader, model.train(), optimizer, loss_fn)
            avg_loss_eval = eval_fn(val_loader, model.eval(), loss_eval)
            save_checkpoint(model, optimizer, f"yolotrain_epoch{epoch}.pth", epoch, avg_loss)
    elif mode == "retrain":
        model, optimizer, start_epoch, start_loss = load_checkpoint(check_point_path, model, optimizer)
        model.eval()
        for epoch in range(start_epoch + 1, epochs):
            # pred sample box by non maxima suppression
            img_sample, label_sample = train_dataset.__getitem__(0)
            # img_shape: 3x448x448
            pred_sample = model(img_sample.unsqueeze(0).to(device))
            # bboxes: [[class_of_box, prob, x, y, w, h]]
            pred_sample = pred_sample.view(1, 7, 7, 30)
            bboxes = non_maxima_suppression(pred_sample, 0.3, 0.5)
            fig, axes = plt.subplots(1,1)
            #imshow
            axes.imshow(img_sample.permute(1,2,0).long())
            if len(bboxes) != 0:
                for box in bboxes:
                    box = [int(i) for i in box]
                    x,y,w,h = box[2], box[3], box[4], box[5]
                    rect = plt.Rectangle((x-w/2,y-h/2),w,h,linewidth=1,edgecolor='r',facecolor='none')
                    axes.add_patch(rect)
                    axes.text(x-w/2, y-h/2, s=decode_label[box[0]], color='r')
            
            plt.show()
            # compute mean_avg_precision

            #
            avg_loss = train_fn(train_loader, model.train(), optimizer, loss_fn)
            avg_loss_eval = eval_fn(val_loader, model.eval(), loss_eval)
            save_checkpoint(model, optimizer, f"yolotrain_epoch{epoch}.pth", epoch, avg_loss)
            
if __name__ == "__main__":
    main("train")