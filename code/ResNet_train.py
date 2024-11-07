import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import torchvision.models.resnet
from img_prepro import *
from ResNet import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

#数据加载
train_data = datasets.ImageFolder(root="D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\Mydata\\train",
           transform=data_transform)
valid_data = datasets.ImageFolder(root="D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\Mydata\\valid",
           transform=data_transform)

train_num = len(train_data)
valid_num = len(valid_data)

train_num = len(train_data)
valid_num = len(valid_data)

PlantDisease_list = train_data.class_to_idx
cla_dict = dict((val, key) for key, val in PlantDisease_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True) #batch_size=4表示加载的数据切分为64个为一组的 shuffle=True表示送人训练的的数据是打乱后送人的，而不是顺序输入
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=64,shuffle=True)


print("using {} images for training, {} images for validation.".format(train_num,
                                                                       valid_num))

# 创建网络模型
resnet = Resnet34()

# 下载预处理权重
model_weight_path = "./resnet34-pre.pth"
#载入模型权重
resnet.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
# for param in net.parameters():
#     param.requires_grad = False

# 输入特征矩阵深度
in_channel = resnet.fc.in_features
resnet.fc = nn.Linear(in_channel, 21)
resnet.to(device)

# 定义损失函数
loss_function = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(resnet.parameters(), lr=0.0001)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_valid_step = 0

epochs = 10
best_acc = 0.0
save_path = './resNet34.pth'
train_steps = len(train_loader)

# 添加tensorboard
writer = SummaryWriter("logs_train")

for epoch in range(epochs):
    # 开始训练
    resnet.train()
    running_loss = 0.0
    print("-------第{}轮训练开始-------".format(epoch + 1))
    for data in train_loader:
        images, labels = data
        optimizer.zero_grad()
        logits = resnet(images.to(device))
        loss = loss_function(logits, labels.to(device))
        loss.backward() # 反向传播
        optimizer.step()

        # 输出总损失
        running_loss += loss.item()

        total_train_step = total_train_step + 1
        if total_train_step%20==0:
            print("训练次数：{},loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # validate
    resnet.eval()
    acc = 0.0
    with torch.no_grad():
        for val_data in valid_loader:
            val_images, val_labels = val_data
            outputs = resnet(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / valid_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))
    total_valid_step = total_valid_step + 1
    writer.add_scalar("val_accuracy", val_accurate, total_valid_step)

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(resnet.state_dict(), save_path)

print('Finished Training')

