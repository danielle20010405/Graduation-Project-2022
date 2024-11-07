import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter("AlexNet_logs")

class AlexNet(nn.Module):
    def __init__(self, num_classes=21,init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(  # Sequential能把一系列的层结构打包成一个新的层结构，当前层结构被定义为提取特征的层结构
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            # 第一层 # input[3, 224, 224]  output[48, 55, 55]，他只用了一半的卷积核（padding=（1，2），计算后是小数，就又一样了）
            nn.ReLU(inplace=True),  # inplace是pytorch通过一种操作增加计算量减少内存占用
            nn.MaxPool2d(kernel_size=3, stride=2),  # 卷积核大小是3，步距是2                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),  # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(  # 分类器
            nn.Dropout(p=0.5),  # dropout的方法上全连接层随机失活（一般放在全裂阶层之间）p值随即失火的比例
            nn.Linear(128 * 6 * 6, 2048),  # linear是全连接层
            nn.ReLU(inplace=True),  # 激活函数
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes) # 输出数据集的类别个数
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)  # 展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

alexnet=AlexNet()

# input=torch.rand([1,3,224,224])
# output=alexnet(input)
#
# writer.add_graph(alexnet,input)
# writer.close()


