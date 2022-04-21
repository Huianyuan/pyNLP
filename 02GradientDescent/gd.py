# coding: utf-8
from tqdm import trange
# 计算(x-2)**2=0
epoch = 1000  # 轮次
lr = 0.05  # 学习率，即更新猜测指的倍率
label = 0  # 真值
x = 5 # 初始值
for e in trange(epoch):
    pre = (x - 2) ** 2  # 预测值
    loss = (pre - label) ** 2  # 损失函数
    delta_x = 2 * (pre - label) * 2 * (x - 2)  # 对损失函数求导
    x = x - delta_x * lr  # 更新下一次的x值，并且对求导值乘以学习率改变它的下降速度

print(x)