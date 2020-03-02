import torch
import torch.nn as nn
import torchvision.models as mdls

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()  

        ''' declare layers used in this network'''
        # first block

        self.resnet = mdls.resnet18(True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1) # 64x64 -> 64x64
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        #self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 64x64 -> 32x32
        
        #second block
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1) # 32x32 -> 32x32
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

        # third block
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # 32x32 -> 32x32
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # 32x32 -> 32x32
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1) # 32x32 -> 32x32
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(16, 9, kernel_size=1, stride=1, padding=0, bias=True) # 32x32 -> 32x32


        # classification
        # self.avgpool = nn.AvgPool2d(16)
        # self.fc = nn.Linear(64, 4)
        # self.avgpool = nn.AvgPool2d(8)
        # # self.fc = nn.Linear(128, 4)

    def forward(self, img):

        x=self.resnet(img)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        return x

