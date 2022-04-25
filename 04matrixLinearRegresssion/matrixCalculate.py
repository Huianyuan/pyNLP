import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

"""
多个特征进行矩阵运算，简化运算量。
下面例子：使用房子的特征 室,厅,卫,面积（平米）,楼层,建成年份 来预测 房价（元/平米）。对应的y为价格，x1-x6为特征，形成一个1X7的矩阵，
数据共有270条，所以X代表270X7的矩阵。进行y=KX+B,式中都是矩阵表示
"""


def get_data(file="上海二手房价.csv"):
    datas = pd.read_csv(file, names=["y", "x1", "x2", "x3", "x4", "x5", "x6"], skiprows=1)  # 读取数据

    y = datas["y"].values.reshape(-1, 1)
    X = datas[[f"x{i}" for i in range(1, 7)]].values

    # z-score: (x-mean_x)/std
    mean_y = np.mean(y)  # 均值
    std_y = np.std(y)  # 方差

    mean_X = np.mean(X, axis=0, keepdims=True)
    std_X = np.std(X, axis=0, keepdims=True)

    y = (y - mean_y) / std_y
    X = (X - mean_X) / std_X

    return X, y, mean_y, std_y, mean_X, std_X


if __name__ == "__main__":
    X, y, mean_y, std_y, mean_X, std_X = get_data()
    K = np.random.random((6, 1))

    epoch = 1000
    lr = 0.01
    b = 0

    for e in range(epoch):
        pre = X @ K + b
        loss = np.sum((pre - y) ** 2) / len(X)

        # G = 2*(pre-y) # 梯度这里的常数项也省略掉，带常数会映射到学习率计算（第45行），导致学习率倍增
        G = (pre - y) / len(X)  # 除以长度，求平均梯度
        delta_K = X.T @ G
        delta_b = np.mean(G)  # 截距这里的梯度就是G，这里取平均，避免过大

        K = K - lr * delta_K
        b = b - lr * delta_b

        print(f"loss:{loss:.3f}")

    while True:
        bedroom = (int(input("请输入卧室数量：")))
        ting = (int(input("请输入客厅数量：")))
        wei = (int(input("请输入卫生间数量：")))
        area = (int(input("请输入面积：")))
        floor = (int(input("请输入楼层：")))
        year = (int(input("请输入年份：")))

        # ap.array(n).reshpae(a,b),依次是生成n个自然数，并以a行b列的数组形式表示，-1表示自动计算当前行或列
        test_x = (np.array([bedroom, ting, wei, area, floor, year]).reshape(1, -1) - mean_X) / std_X

        p = test_x @ K + b
        print("房价为：", p * std_y + mean_y)
