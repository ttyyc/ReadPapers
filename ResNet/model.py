import torch
import torch.nn as nn

class Residual1(nn.Module):
    '''residual block for 18/34 layer resnet'''
    def __init__(self, in_channels: int, out_channels: int, downsampling: bool):
        super().__init__()
        if downsampling:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2)
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,stride=2)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            self.shortcut = nn.Identity()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_copy = x
        x = self.act(self.bn(self.conv1(x)))
        x = self.bn(self.conv2(x))
        x_copy = self.shortcut(x_copy)
        x = self.act(x + x_copy)
        return x
    
class Residual2(nn.Module):
    '''residual block for 50/101/152 layer resnet'''
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, downsampling: bool):
        super().__init__()
        if downsampling:
            self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=2)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        else:
            self.conv1 = nn.Conv2d(out_channels, hidden_channels, kernel_size=1, stride=1)
            self.shortcut = nn.Identity()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.act = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_copy = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x_copy = self.shortcut(x_copy)
        x = self.act(x+x_copy)
        return x
    

class ResNet(nn.Module):
    def __init__(self, architecture: list=[18,2,2,2,2], hidden_num: int=2048, num_classes: int=1000):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.gap = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(in_features=hidden_num, out_features=num_classes)
        self.softmax = nn.Softmax(1)
        if architecture[0] < 50:
            self.block = self.stack_blocks(Residual1, architecture)
        else:
            self.block = self.stack_blocks(Residual2, architecture)

    
    def stack_blocks(self, block, architecture):
        if(block.__name__=='Residual1'):
            blocks = nn.ModuleList([Residual1(64*(2**(j-1)), 64*(2**(j-1)),True) if(j==1 and i==0) else Residual1(64*(2**(j-2)), 64*(2**(j-1)),True) if(j!=1 and i == 0) else Residual1(64*(2**(j-1)), 64*(2**(j-1)),False) for j in range(1,5) for i in range(architecture[j])])
        else:
            blocks = nn.ModuleList([Residual2(64*(2**(j-1)),64*(2**(j-1)),256*(2**(j-1)),True) if(j==1 and i == 0) else Residual2(256*(2**(j-2)), 64*(2**(j-1)),256*(2**(j-1)),True) if(j!=1 and i == 0) else Residual2(256*(2**(j-1)), 64*(2**(j-1)),256*(2**(j-1)),False) for j in range(1,5) for i in range(architecture[j])])
        return blocks

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.maxpool(x)
        for block in self.block:
            x = block(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        x = self.softmax(x)
        return x


def resnet18(num_classes: int=1000):
    '''build 18 layer resnet'''
    return ResNet([18,2,2,2,2], 512, num_classes)

def resnet34(num_classes: int=1000):
    '''build 34 layer resnet'''
    return ResNet([34,3,4,6,3], 512, num_classes)

def resnet50(num_classes: int=1000):
    '''build 50 layer resnet'''
    return ResNet([50,3,4,6,3], 2048, num_classes)

def resnet101(num_classes: int=1000):
    '''build 101 layer resnet'''
    return ResNet([101,3,4,23,3], 2048, num_classes)

def resnet152(num_classes: int=1000):
    '''build 152 layer resnet'''
    return ResNet([152,3,8,36,3], 2048, num_classes)

if __name__=='__main__':
    b1 = resnet152()
    print(b1)
    x = torch.ones((1,3,224,224))
    out = b1(x)
    print(out.shape)

