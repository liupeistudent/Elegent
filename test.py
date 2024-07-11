import os.path
import torch
from read_data import  *
from VGG16 import *
from torchvision.utils import save_image
from torch.utils.data import  DataLoader
# from utils import *
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #判断使用

net = VGG16_Net().cpu()
weights = 'params/VGG16.pth'

if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('successfully')

else:
    print('no')
test_path = 'D:\\learn_python\\minist_learning\\MNIST\\MNIST\\rawtest.txt'
_input = MyDataset('D:\\learn_python\\minist_learning\\MNIST\\MNIST\\rawtest.txt')
data_loader = DataLoader(MyDataset(test_path),batch_size=1,shuffle=False)
for i, (img, label) in enumerate(data_loader):
    #img, label = img.to(device), label.to(device)
    # print(label.shape)  ###
    # 梯度清零
    #opt.zero_grad()
    # 在模型上前向传播和反向传播
    outputs = net(img)
    index = torch.argmax(outputs)
    print(index,label)

    #train_loss = loss_fun(outputs, label)
    #train_loss.backward()
    #opt.step()


#print(data_loader)
# _input = input('please input image path')
# img = Image.open(_input).convert('RGB')
# #img = transform(img)
# img_data = transform(img)
# out = net(img_data)
# print(out)

