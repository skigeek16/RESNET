import torch 
import torch.nn as nn

class block(nn.Module):
    def __init__(self,inchannel,outchannels,identity_downsample=None,stride=1):
        super(block,self).__init__()
        self.expansion=4
        self.conv1=nn.Conv2d(inchannel,outchannels,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(outchannels)
        self.conv2=nn.Conv2d(outchannels,outchannels,kernel_size=3,stride=stride,padding=1)
        self.bn2=nn.BatchNorm2d(outchannels)
        self.conv3=nn.Conv2d(outchannels,outchannels*self.expansion,kernel_size=1,stride=1,padding=0)
        self.bn3=nn.BatchNorm2d(outchannels*self.expansion)
        self.relu=nn.ReLU()
        self.identity_downsample=identity_downsample
    def forward(self,x):
        identity=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.bn3(x)

        if self.identity_downsample is not None:
            identity=self.identity_downsample(identity)

        x+=identity
        x=self.relu(x)

class Resnet(nn.Module):
    def __init__(self,block,layers,imagechannels,numclasses ):
        super(Resnet,self).__init__()
        self.in_channel=64
        self.conv1=nn.Conv2d(imagechannels,64,kernel_size=7,stride=2,padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1=self._make_layer(block,layers[0],outchannels=64,stride=1)
        self.layer2=self._make_layer(block,layers[1],outchannels=128,stride=2)
        self.layer3=self._make_layer(block,layers[2],outchannels=256,stride=2)
        self.layer4=self._make_layer(block,layers[3],outchannels=512,stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*4,numclasses)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        return x
    def _make_layer(self,block,num_res,outchannels,stride):
        identity_downsample=None
        layers=[]

        if stride!=1 or self.in_channel!=outchannels*4:
            identity_downsample=nn.Sequential(nn.Conv2d(self.in_channel,outchannels*4,kernel_size=1,stride=stride)) 

        layers.append(block(self.in_channel,outchannels,identity_downsample,stride))
        self.in_channel=outchannels*4

        for i in range(num_res-1):
            layers.append(block(self.in_channel,outchannels))
        return nn.Sequential(*layers)


def Resnet50(img_channels=3,num_classes=1000):
    return Resnet(block,[3,4,6,3],img_channels,num_classes)

        

        
    