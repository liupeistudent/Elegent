from torch import  nn,optim
import torch
from torch.utils.data import  DataLoader
from  read_data import *

from VGG16 import  *

# https://github.com/liupeistudent/VGG16.git
from torchvision.utils import save_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #判断使用
weight_path = 'params/VGG16.pth'  #权重存储位置
data_path = r'D:\learn_python\minist_learning\MNIST\MNIST\rawtrain.txt'
#save_path = 'train_image'
# Pycharm编辑器是一个非常流行的工
#
if __name__ == '__main__':
    data_loader = DataLoader(MyDataset(data_path),batch_size=4,shuffle=False)
    #print(data_loader)
    net = VGG16_Net().to(device)
    # if os.path.exists(weight_path):
    #     net.load_state_dict(torch.load(weight_path))
    #     print('successful weight')
    # else:
    #     print('not successful weight')


    opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    loss_fun =  nn.CrossEntropyLoss()

    epoch = 1
    while True:

        for i,(img,label) in enumerate(data_loader):
            img, label = img.to(device), label.to(device)
            #print(label)  ###
            #梯度清零
            opt.zero_grad()
            #在模型上前向传播和反向传播
            outputs = net(img)
            print(outputs)
            print(label)  ###
            train_loss =loss_fun(outputs, label)
            train_loss.backward()
            opt.step()


            #running_loss += train_loss.item()
            # if i % 100 == 99:
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 100))
            #     running_loss = 0.0

            if i % 5 == 0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i % 50 == 0:
                torch.save(net.state_dict(), weight_path)

            # _image = image[0]
            # _segment_image = segment_image[0]
            # _out_image = out_image[0]
            #
            # img = torch.stack([_image, _segment_image, _out_image], dim=0)
            # save_image(img, f'{save_path}/{i}.png')

        epoch += 1










