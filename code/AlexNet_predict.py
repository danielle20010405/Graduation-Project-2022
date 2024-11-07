import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from img_prepro import *
from AlexNet import *


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据
img_path = "D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\Mydata\\test\\TomatoEarlyBlight6.JPG"
img = Image.open(img_path)
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# 扩充维度
img = torch.unsqueeze(img, dim=0)

# 读取索引对应的类别名称
try:
    json_file = open("D:\project\PyCharm\learn_pytorch\PlantDisease\class_indices.json",'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 创建模型
alexnet = AlexNet(num_classes=21).to(device)

# 载入模型权重
alexnet_weights_path = "D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\AlexNet.pth"
alexnet.load_state_dict(torch.load(alexnet_weights_path))
alexnet.eval()
with torch.no_grad():
    # 预测
    output = torch.squeeze(alexnet(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)
    predict_cla = torch.argmax(predict).numpy()

print(class_indict[str(predict_cla)],predict[predict_cla].item())

# plt.show()



