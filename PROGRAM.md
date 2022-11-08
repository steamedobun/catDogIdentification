# 项目实战1

## 1.1项目说明

### 1.1.1项目名称

使用pytorch框架编写利用VGG16网络的猫狗分类程序

### 1.1.2参考教程

https://www.bilibili.com/video/BV1X3411N7aj

## 1.2项目详解

### 1.2.1项目流程

#### 1.加载数据集

加载猫狗数据集，文件夹train下cat、dog文件夹

#### 2.生成txt文件标记

txt.py运行得到cls_train.txt文件

#### 3.数据处理

data.py

#### 4.搭建三层神经网络

net.py

#### 5.训练数据

main.py

其中利用到搭建的神经网络net.py

#### 6.预测

predict.py

### 1.2.2项目代码

#### 1.txt.py

```python
import os
from os import getcwd

classes = ['cat', 'dog', 'mushroom']  # 定义三个类型猫、狗、蘑菇
sets = ['train']

if __name__ == '__main__':
    wd = getcwd()  # 在Python中可以使用os.getcwd()函数获得当前的路径。
    for se in sets:
        list_file = open('cls_' + se + '.txt', 'w')

        datasets_path = se
        types_name = os.listdir(datasets_path)  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for type_name in types_name:
            if type_name not in classes:
                continue
            cls_id = classes.index(type_name)  # 输出0-1
            photos_path = os.path.join(datasets_path, type_name)  # 依次读取数据集内容
            photos_name = os.listdir(photos_path)
            for photo_name in photos_name:
                _, postfix = os.path.splitext(photo_name)  # 该函数用于分离文件名与拓展名
                if postfix not in ['.jpg', '.png', '.jpeg']:
                    continue
                # 依次写入文件
                list_file.write(str(cls_id) + ';' + '%s/%s' % (wd, os.path.join(photos_path, photo_name)))
                list_file.write('\n')
        list_file.close()

```

写出文件格式如下：

> ```
> 0;F:\OpencvCode\catDogRecognition\text\VGG16-main\VGGNet/train\cat\cat.0.jpg
> 0;F:\OpencvCode\catDogRecognition\text\VGG16-main\VGGNet/train\cat\cat.1.jpg
> 0;F:\OpencvCode\catDogRecognition\text\VGG16-main\VGGNet/train\cat\cat.10.jpg
> 0;F:\OpencvCode\catDogRecognition\text\VGG16-main\VGGNet/train\cat\cat.100.jpg
> ```

#### 2.data.py

用于对图片大小形式修改(因为传入的数据中图片格式不唯一)

```python
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image


def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[-2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, inpt_shape, random=True):
        self.annotation_lines = annotation_lines
        self.input_shape = inpt_shape
        self.random = random

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
        image = Image.open(annotation_path)
        image = self.get_random_data(image, self.input_shape, random=self.random)
        image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
        y = int(self.annotation_lines[index].split(';')[0])
        return image, y

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a)+a

    def get_random_data(self, image, inpt_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):

        image = cvtColor(image)
        iw, ih = image.size
        h, w = inpt_shape
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))

            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            return image_data
        new_ar = w/h*self.rand(1-jitter, 1+jitter)/self.rand(1-jitter, 1+jitter)
        scale = self.rand(.75, 1.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # 翻转图像
        flip = self.rand() < .5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-15, 15)
            a, b = w/2, h/2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])
        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)  # 颜色空间转换
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data
```

#### 3.net.py

搭建的网络vgg16

```python
import torch
import torch.nn as nn

from torch.hub import load_state_dict_from_url


model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",

}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, dropout=0.5):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],

}


def vgg16(pretrained=True, progress=True,num_classes=2):
    model = VGG(make_layers(cfgs['D']))
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir='./model', progress=progress)
        model.load_state_dict(state_dict)
    if num_classes != 1000:
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
    return model


if __name__ == '__main__':
    in_data = torch.ones(1, 3, 224, 224)  # 与之对应的四维全1矩阵
    net = vgg16(pretrained=True, progress=True, num_classes=2)
    out = net(in_data)
    print(out)  # 查看输出

```

通过地址获取模型https://download.pytorch.org/models/vgg16-397923af.pth

```python
model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",

}
```

#### 4.main.py

训练数据代码

1.加载数据集、数据标记(统一数据图片大小)

2.构建vgg16网络

3.选择优化器

4.调整学习率 lr

5.训练(通过for循环epochs次)

<!--深度学习中的epochs，batch_size，iterations-->

<!--epochs被定义为向前和向后传播中所有批次的单次训练迭代。这意味着1个周期是整个输入数据的单次向前和向后传递。简单说，epochs指的就是训练过程中数据将被“轮”多少次。-->

<!--iterations（迭代）：每一次迭代都是一次权重更新，每一次权重更新需要batch_size个数据进行Forward运算得到损失函数，再BP算法更新参数。1个iteration等于使用batchsize个样本训练一次。-->

<!--batch深度学习的优化算法，梯度下降。每次的参数更新有两种方式。第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。第二种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。基本上现在的梯度下降都是基于mini-batch的，所以深度学习框架的函数中经常会出现batch_size，就是指这个。-->

<!--具体的计算公式为：one **epoch** = numbers of **iterations** = N = 训练样本的数量 / **batch_size**-->

```python
import torch
import torch.nn as nn
from net import vgg16
from torch.utils.data import DataLoader
from data import *

'''数据集'''
annotation_path = 'cls_train.txt'
with open(annotation_path, 'r') as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)  # 打乱数据
np.random.seed(None)
num_val = int(len(lines)*0.1)
num_train = len(lines)-num_val
# 输入图像大小
input_shape = [224, 224]
train_data = DataGenerator(lines[:num_train], input_shape, True)
val_data = DataGenerator(lines[num_train:], input_shape, False)
val_len = len(val_data)

"""加载数据"""
gen_train = DataLoader(train_data, batch_size=4)
gen_test = DataLoader(val_data, batch_size=4)

'''构建网络'''
device = torch.device('cuda'if torch.cuda.is_available() else "cpu")
net = vgg16(pretrained=True, progress=True, num_classes=2)
net.to(device)

'''选择优化器和学习率的调整方法'''
lr = 0.0001
optim = torch.optim.Adam(net.parameters(), lr=lr)
sculer = torch.optim.lr_scheduler.StepLR(optim, step_size=1)

'''训练'''
epochs = 15
for epoch in range(epochs):
    total_train = 0
    for data in gen_train:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
        optim.zero_grad()
        output = net(img)
        train_loss = nn.CrossEntropyLoss()(output, label).to(device)
        train_loss.backward()
        optim.step()
        total_train += train_loss
    total_test = 0
    total_accuracy = 0
    for data in gen_test:
        img, label = data
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            optim.zero_grad()
            out = net(img)
            test_loss = nn.CrossEntropyLoss()(out, label).to(device)
            total_test += test_loss
            accuracy = (out.argmax(1) == label).sum()
            total_accuracy += accuracy
    print("训练集上的损失：{}".format(total_train))
    print("测试集上的损失：{}".format(total_test))
    print("测试集上的精度：{:.1%}".format(total_accuracy/val_len))
    # torch.save(net,"dogandcat.{}.pt".format(epoch+1))
    torch.save(net.state_dict(), "Adogandcat.{}.pth".format(epoch+1))
    print("模型已保存")

```

#### 5.predict.py

预测代码

1.处理预测图片

2.加载vgg16网络

3.通过matplotlib.pyplot库绘图

```python
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from net import vgg16

img_pth = './img/cat.3.jpg'
img = Image.open(img_pth)
'''处理图片'''
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
image = transform(img)
'''加载网络'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = vgg16()
model = torch.load("./Adogandcat.1.pth", map_location=device)
net.load_state_dict(model)
net.eval()
image = torch.reshape(image, (1, 3, 224, 224))
with torch.no_grad():
    out = net(image)
out = F.softmax(out, dim=1)
out = out.data.cpu().numpy()
print(out)
a = int(out.argmax(1))
plt.figure()
list = ["0", '1']
# plt.suptitle("Classes:{}:{:.1%}".format(list[a], out[0, a]))
plt.suptitle("cat{:.1%}, dog{:.1%}".format(out[0, 0], out[0, 1]))
plt.imshow(img)
plt.show()
```

### 1.2.3项目成果

预测结果(训练遍数为3，精度偏低)

![image-20221022192824251](图表/image-20221022192824251.png)

## 1.3项目总结

### 1.3.1实现思路

1.首先导入需要的包，如pytorch：import torch，和其中一些必要的库 import torch.nn as nn；from torch.autograd import Variable ......

2.定义自己的网络（vgg16）

3.训练模型：1.初始化模型，2.选择优化器以及优化算法，3.选择损失函数，这里选择了交叉熵，4.对每一个batch里的数据，先将它们转成能被GPU计算的类型，5.梯度清零、前向传播、计算误差、反向传播、更新参数

4.测试

### 1.3.2项目感想

通过项目学习到

1.如何通过步骤搭建网络，借助框架实现训练和预测。但前提在于导入所需库实现功能。

2.影响预测精度的有epochs，batch_size，iterations等参数的调整。

3.torch.nn，nn模块下的Module类，组件类，neture network
   tensor，巩固tensor的方法
   torch.utils.data里面DataLoader的用法
   torchvision里面transforms的用法
   torchvision.datasets里面ImageFolder的用法
