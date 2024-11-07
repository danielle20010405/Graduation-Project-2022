import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from img_prepro import *
from ResNet import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据图像
img_path = "D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\Mydata\\test\\TomatoYellowCurlVirus3.JPG"
img = Image.open(img_path)
# plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open("D:\project\PyCharm\learn_pytorch\PlantDisease\class_indices.json",'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 创建模型
resnet34 = Resnet34(num_classes=21).to(device)

# 加载模型权重
weights_path = "D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\resNet34.pth"
assert os.path.exists(weights_path),"file:'{}' dose not exist. ".format(weights_path)
resnet34.load_state_dict(torch.load(weights_path, map_location=device))

# 预测
resnet34.eval()
with torch.no_grad():
    # predict class
    output = torch.squeeze(resnet34(img.to(device))).cpu()
    predict = torch.softmax(output, dim=0)  # softmax函数实现多分类
    predict_cla = torch.argmax(predict).numpy()

print_res = "class:{}   prot:{:.3}".format(class_indict[str(predict_cla)],
                                           predict[predict_cla].numpy())

plt.title(print_res)
print(print_res)

#plt.show()


