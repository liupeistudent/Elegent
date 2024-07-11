from torch import nn
#from torch.nn import functional as F
import torch
import  numpy as np
class VGG16_Net(nn.Module):
    def __init__(self):
        super(VGG16_Net,self).__init__()
        self.layer1 = nn.Sequential(#第一层（卷积卷积池化）
            nn.Conv2d(3, 64, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(64),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(64, 64, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(64),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),

            #池化层 改变数据的大小不改变通道数
        )

        self.layer2 = nn.Sequential(

            nn.Conv2d(64, 128, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(128),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(128, 128, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(128),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(256, 256, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(256),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(256, 256, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(256),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(512),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(512),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(512),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数

            nn.Conv2d(512, 512, 3, 1, 1, padding_mode='reflect', bias=False),
            # 卷积核为3，步长为1，边缘填充为1，填充模式为反射填充，无偏置项的二维卷积层
            nn.BatchNorm2d(512),  # 将数据归一化处理
            nn.Dropout2d(0.3),  # 以0.3的概率随机将一些神经元失活
            nn.LeakyReLU(),  # 激活函数
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc6 = nn.Linear(512 * 7* 7, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout6 = nn.Dropout()
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout7 = nn.Dropout()
        self.fc8 = nn.Linear(4096, 10)


    def forward(self,x):
        print(x.shape)

        x = self.layer1(x)
        print(x.shape)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        #x = x.view(x.size(0), -1)
        # print(x)
        x = x.view(-1, 7* 7 * 512)
        # print(x)


        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc8(x)
        return x
if __name__ == '__main__':
    x = torch.randn((1,3,224,224))
    net = VGG16_Net()


    print(net(x))