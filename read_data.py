from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torchvision import transforms
import torch


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),

])
path = r'D:\learn_python\minist_learning\MNIST\MNIST\rawtrain.txt'
class MyDataset(Dataset):
    def __init__(self,path):
        self.path = path
        #self.name =   os.listdir(os.path.join(path,'train'))
        fh = open(path)
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1].split("(")[1].split(")")[0])))
            self.imgs = imgs
    def __len__(self):
        return (len(self.imgs))


    def __getitem__(self, index):
        fn,label = self.imgs[index]
        img =Image.open(fn).convert('RGB')
        img = transform(img)
        label = torch.tensor(label)

        return img,label

if __name__ == '__main__':
    data = MyDataset(path)
    print(data[0][0].shape)
