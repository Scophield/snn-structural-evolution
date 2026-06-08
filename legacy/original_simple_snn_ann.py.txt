import os
import random

import torch
import torchvision.datasets

from torch import nn, randperm
from torch.cuda import amp
from torch.cuda.amp import autocast
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, AvgPool2d, LPPool2d, Sequential, ReLU, BatchNorm2d
from torch.utils.data import DataLoader, random_split

from torchvision.transforms import transforms, InterpolationMode
from tqdm import tqdm

from Alexnet_spiking import Spike_alex
# from TGRS import Spike
# from model import Spikformer


from lenet_spiking import Spike

class sigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = 4.0
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1. - sgax) * sgax * ctx.alpha

        return grad_x, None
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        self.img_size = 32
        self.num_cls = 10
        self.num_steps = 4
        self.spike_fn = sigmoid.apply
        self.lin1 = nn.Linear(28*28, 10)
        self.lin2 = nn.Linear(10, 10)
    def forward(self, imgs):


        x=imgs.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)
        def mlifnodelayer(x_seq):
            spike_seq = []
            for t in range(x_seq.shape[0]):
                if t==0 :
                    mem =x_seq[t]
                else:
                    mem = mem + x_seq[t]
                x = self.spike_fn(mem)
                spike_seq.append(x.unsqueeze(0))
                mem = mem - x
            spike_seq = torch.cat(spike_seq, 0)

            return spike_seq

        # T,B,C,H,W=x.shape
        # B 1 28 28  ,T B 1 28 28  ,T B 1*28*28
        x = nn.Flatten(2)(x)

        x = self.lin1(x)# T B 10

        x = mlifnodelayer(x)# T B 10   0/1
        x = self.lin2(x)# T B 10  FLOAT

        return x.mean(0)#  B 10

class stage0(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1=Linear(28*28,10)
        self.lin2 = Linear(10, 10)
    def forward(self,x):
        fl = Flatten()

        relu=nn.ReLU()
        x=fl(x)
        x = self.lin1(x)
        x = relu(x)
        x = self.lin2(x)
        return  x

class stage1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin1=Linear(28*28,10)
        self.lin2 = Linear(10, 10)
        self.spike_fn = sigmoid.apply
    def forward(self,x):
        fl = Flatten()

        relu=nn.ReLU()
        x=fl(x)
        x = self.lin1(x)
        x = self.spike_fn(x-1)#此处可以忽略self.spike_fn(x_seq[t]-1)具体实现细节，将其等价于torch.where(x>1,1,0)即可
        x = self.lin2(x)
        return  x
class stage2(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        self.img_size = 32
        self.num_cls = 10
        self.num_steps = 4
        self.spike_fn = sigmoid.apply
        self.lin1 = nn.Linear(28*28, 10)
        self.lin2 = nn.Linear(10, 10)
    def forward(self, imgs):


        x=imgs.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)#此处需注意，这里使用直接编码，约等于不编码。将图像扩展致时间步倍数后放入网络。
        def mlifnodelayer(x_seq):
            spike_seq = []
            for t in range(x_seq.shape[0]):
                x = self.spike_fn(x_seq[t]-1)#此处可以忽略self.spike_fn(x_seq[t]-1)具体实现细节，将其等价于torch.where(x>1,1,0)即可
                spike_seq.append(x.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            return spike_seq

        x = nn.Flatten(2)(x)
        x = self.lin1(x)# T B 10
        x = mlifnodelayer(x)# 需要注意，此处mlifnodelayer的使用等价于self.spike_fn(x-1).但为了更好的向靠后阶段拓展，采用了此编写方式
        x = self.lin2(x)# T B 10  FLOAT

        return x.mean(0)#  B 10
class stage3(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        self.img_size = 32
        self.num_cls = 10
        self.num_steps = 4
        self.spike_fn = sigmoid.apply
        self.lin1 = nn.Linear(28*28, 10)
        self.lin2 = nn.Linear(10, 10)
    def forward(self, imgs):


        x=imgs.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)#此处需注意，这里使用直接编码，约等于不编码。将图像扩展致时间步倍数后放入网络。
        def mlifnodelayer(x_seq):
            spike_seq = []
            for t in range(x_seq.shape[0]):
                if t==0 :#需注意此IF函数里是第三阶段的主要特征
                    mem =x_seq[t]
                else:
                    mem = mem + x_seq[t]
	x = self.spike_fn(x_seq[t]-1)	#此处可以忽略self.spike_fn(x_seq[t]-1)具体实现细节，将其等价于torch.where(x>1,1,0)即可
                spike_seq.append(x.unsqueeze(0))
            spike_seq = torch.cat(spike_seq, 0)
            return spike_seq

        x = nn.Flatten(2)(x)
        x = self.lin1(x)# T B 10
        x = mlifnodelayer(x)# 需要注意，此处mlifnodelayer的使用等价于self.spike_fn(x-1).但为了更好的向靠后阶段拓展，采用了此编写方式
        x = self.lin2(x)# T B 10  FLOAT

        return x.mean(0)#  B 10

class stage4(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()

        self.img_size = 32
        self.num_cls = 10
        self.num_steps = 4
        self.spike_fn = sigmoid.apply
        self.lin1 = nn.Linear(28*28, 10)
        self.lin2 = nn.Linear(10, 10)
    def forward(self, imgs):


        x=imgs.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1)#此处需注意，这里使用直接编码，约等于不编码。将图像扩展致时间步倍数后放入网络。
        def mlifnodelayer(x_seq):
            spike_seq = []
            for t in range(x_seq.shape[0]):
                if t==0 :#需注意此IF函数里是第三阶段的主要特征
                    mem =x_seq[t]
                else:
                    mem = mem + x_seq[t]
	x = self.spike_fn(x_seq[t]-1)	#此处可以忽略self.spike_fn(x_seq[t]-1)具体实现细节，将其等价于torch.where(x>1,1,0)即可
                spike_seq.append(x.unsqueeze(0))
	mem = mem - x#需要注意此处为第四阶段主要特征
            spike_seq = torch.cat(spike_seq, 0)
            return spike_seq

        x = nn.Flatten(2)(x)
        x = self.lin1(x)# T B 10
        x = mlifnodelayer(x)# 需要注意，此处mlifnodelayer的使用等价于self.spike_fn(x-1).但为了更好的向靠后阶段拓展，采用了此编写方式
        x = self.lin2(x)# T B 10  FLOAT

        return x.mean(0)#  B 10
import numpy as np


trainset=torchvision.datasets.MNIST(root="/home/ps/hhx/lenet/data",train=True,transform=transforms.Compose([
        transforms.ToTensor(),
    ]),download=True )

testset=torchvision.datasets.MNIST(root="/home/ps/hhx/lenet/data",train=False,transform=transforms.Compose([
        transforms.ToTensor(),
    ]),download=True)


test_data_size=len(testset)
train_data_size=len(trainset )

if __name__ == '__main__':
    def seed_all(seed=1029):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_all(35)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True,num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False,num_workers=0)

    spike=ANN().cuda(0)
    # spike=SNN().cuda(0)
    # print(spike.T)
    # scaler = amp.GradScaler()
    n_parameters = sum(p.numel() for p in spike.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")

    loss_fn=nn.CrossEntropyLoss()

    loss_fn=loss_fn.cuda(0)
    learning_rate=0.005
    epoch = 100
    print("epoch{}".format(epoch))
    optimizer=torch.optim.Adam(spike.parameters(),lr=learning_rate)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epoch,eta_min=2e-4)


    step = 0
    test_step=0
    max=MaxPool2d(2)
    avg=AvgPool2d(2)
    mix_off=0

    best_ac = 0
    for i in range(epoch):




        print("第{}轮训练开始了".format(i+1))
        loss_b=0
        total_accuracy1=0

        if i == mix_off:
            print("mix off")

        lrl = [param_group['lr'] for param_group in optimizer.param_groups]
        print("lrl{}".format(lrl))
        progress_bar = tqdm(total=len(train_loader))
        if 1:
            for data in train_loader:
                imgs, targets = data
                targets = targets.cuda(0)
                imgs=imgs.cuda(0)
                outputs= spike(imgs)
                loss = loss_fn(outputs, targets)

                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy1 = total_accuracy1 + accuracy
                loss_b=loss_b+loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(1)
            progress_bar.close()

            sch.step()

        print("这是第{}个epoch训练的损失".format(i+1),loss_b)
        print("训练集正确率{}".format(total_accuracy1/train_data_size))


        total_test_lost1=0

        total_accuracy_test1=0


        with torch.no_grad():

            n = 0
            for data in test_loader:
                imgs,targets=data
                targets = targets.cuda(0)
                imgs = imgs.cuda(0)
                output= spike(imgs)

                loss = loss_fn(output, targets)

                total_test_lost1=total_test_lost1+loss

                accuracy = (output.argmax(1) == targets).sum()
                total_accuracy_test1 = total_accuracy_test1 + accuracy

            ac=total_accuracy_test1 / test_data_size

            if ac>best_ac:
                best_ac=ac
                torch.save(spike, "combat_best.pth")

            print("整体测试集上的loss：{}".format(total_test_lost1))
            print("整体测试集上的正确率{}".format(ac))
            print("best{}".format(best_ac))











