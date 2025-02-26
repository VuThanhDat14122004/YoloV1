import torch
import torch.nn as nn


architecture_conf = [
    #tuple(kernel_size, channel_out, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    #list of tuple with num of repeat in the last
    [(1,256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
] 

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels,**kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)
    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_conf
        self.in_channels = in_channels
        self.convs = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fc_layers(**kwargs)

    def forward(self, x):
        x = self.convs(x)
        return self.fcs(torch.flatten(x, start_dim=1))
    
    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for ar in architecture:
            if type(ar) == tuple:
                layers += [CNNBlock(in_channels,
                                   ar[1],
                                   kernel_size=ar[0],
                                   stride=ar[2],
                                   padding=ar[3])
                          ]
                in_channels=ar[1]
            elif type(ar) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif type(ar) == list:
                conv1 = ar[0]
                conv2 = ar[1]
                repeat = ar[2]
                for _ in range(repeat):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride=conv1[2],
                            padding=conv1[3]
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size = conv2[0],
                            stride=conv2[2],
                            padding=conv2[3]
                        )
                    ]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def create_fc_layers(self, split_size, num_boxes, num_classes):
        '''
            S=7, B=2, C=20
            split image to SxS, pred B(2) type of box is tall, wide
            20 class of data
        '''
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 400),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(400, S*S*(C+B*5)),# (S*S*30)=1470, C+B*5 = 30
        )
