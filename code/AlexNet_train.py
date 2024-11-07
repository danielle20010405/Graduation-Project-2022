import json
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from img_prepro import *
from AlexNet import *

#数据加载
train_data = datasets.ImageFolder(root="D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\Mydata\\train",
           transform=data_transform)
valid_data = datasets.ImageFolder(root="D:\\project\\PyCharm\\learn_pytorch\\PlantDisease\\Mydata\\valid",
           transform=data_transform)

train_num = len(train_data)
valid_num = len(valid_data)

PlantDisease_list = train_data.class_to_idx
cla_dict = dict((val, key) for key, val in PlantDisease_list.items())
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True) #batch_size=4表示加载的数据切分为64个为一组的 shuffle=True表示送人训练的的数据是打乱后送人的，而不是顺序输入
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=64,shuffle=True) #batch_size=4表示加载的数据切分为64个为一组的 shuffle=True表示送人训练的的数据是打乱后送人的，而不是顺序输入

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 创建网络模型
alexnet = AlexNet(num_classes=21, init_weights=True).to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.0002
optimier = torch.optim.Adam(alexnet.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_valid_step = 0
# 训练轮数
epochs = 10  #设置训练轮数

save_path = './AlexNet.pth'
best_acc = 0.0
train_steps = len(train_loader)

# 添加tensorboard
writer = SummaryWriter("logs_train")

# 训练步骤开始
for epoch in range(epochs):
    alexnet.train()
    running_loss = 0.0
    print("-------第{}轮训练开始-------".format(epoch+1))

    for data in train_loader:
        imgs,labels = data
        output = alexnet(imgs.to(device))
        loss = loss_fn(output,labels)

        # 优化器优化模型
        optimier.zero_grad()
        loss.backward()
        optimier.step()

        # 损失
        running_loss += loss.item()

        total_train_step=total_train_step+1
        if total_train_step%100==0:
            print("训练次数：{},loss:{}".format(total_train_step,loss))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # validate
    alexnet.eval()
    acc = 0.0  # 计算准确率 number / epoch

    with torch.no_grad():
        for val_data in valid_loader:
            val_images, val_labels = val_data
            outputs = alexnet(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

    val_accurate = acc / valid_num
    print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, val_accurate))
    total_valid_step = total_valid_step + 1
    writer.add_scalar("val_accuracy",val_accurate, total_valid_step)

    if val_accurate > best_acc:
        best_acc = val_accurate
        torch.save(alexnet.state_dict(), save_path)
        print("best_acc:{}".format(best_acc))

writer.close()


print('Finished Training')