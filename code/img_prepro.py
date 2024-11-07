from torchvision.transforms import transforms

data_transform = transforms.Compose([
 transforms.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为32像素
 # transforms.CenterCrop(32), # 从图片中间切出32*32的图片
 transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
 transforms.Normalize(mean=[0.3867699, 0.4292917, 0.39402938], std=[0.18648247, 0.16961637, 0.21869354]) # 标准化至[-1, 1]，规定均值和标准差
])