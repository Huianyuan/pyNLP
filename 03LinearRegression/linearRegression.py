# 线性回归（预测房价）
# y=kx+b，已知x，y确定k和b（训练），给定初始值，不断迭代优化
# pre=kx+b, loss=(pre-lable)^2=(kx-b-lable)^2
# 对loss求其对k和b的偏导：D和d
# 然后更新k=k-D*lr，b=b-d*lr  lr为学习率

import numpy as np

years = np.array([i for i in range(2000, 2022)])  # 年份2000~2021
years = (years - 2022) / 22  # Layer_normalizer,batch_normalizer  两种分布，前者NLP较多，后者图像
prices = np.array([10000, 11000, 12000, 13000, 14000, 12000, 13000, 16000, 18000,20000, 19000,
                   22000, 23000, 26000, 35000, 30000, 40000, 45000, 52000, 50000, 60000])/60000

epoch = 1000
k = 1
b = 1
# lr = 0.0000001  # 数据规划问题导致学习率至少小数点后6位起步。第10行代码注释掉 则用这个学习率
lr =0.1
for e in range(epoch):
    for x, lable in zip(years, prices):
        pre = k * x + b
        loss = (pre - lable) ** 2

        delta_k = 2 * (k * x + b - lable) * x
        delta_b = 2 * (k * x + b - lable)

        k = k - delta_k * lr
        b = b - delta_b * lr

print(f"k={k},b={b}")
while True:
    # year = float(input("请输入年份："))
    year = (float(input("请输入年份："))-2022)/22
    # price = k * year + b
    price = (k * year + b)*60000
    print(f"预测房价{price}")
