from PIL import Image
#将数据集不同大小的图片转换为同一大小的数据
#等比缩放代码
def keep_image_size_open(path,size=(256,256)):  #定义一个变换图像形式的函数
    img=Image.open(path)  #
    temp= max(img.size)
    mask= Image.new('RGB',(temp,temp),(0,0,0)) #模式，大小，颜色
    mask.paste(img,(0,0))  #将img贴到mask上面去，位置从左上角开始
    mask = mask.resize(size) #将mask重构为256*256
    return mask