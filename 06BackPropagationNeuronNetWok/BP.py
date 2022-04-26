import numpy as np
import struct  # 二进制数据类型的转换
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TKAgg")


def load_labels(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    return np.asanyarray(bytearray(data[8:]), dtype=np.int32)


def load_images(file):  # 加载数据
    with open(file, "rb") as f:
        data = f.read()
    magic_number, num_items, rows, cols = struct.unpack(">iiii", data[:16])
    return np.asanyarray(bytearray(data[16:]), dtype=np.uint8).reshape(num_items, -1)


def make_one_hot(labels, class_num=10):  # onehot编码转换
    result = np.zeros((len(labels), class_num))
    for index, lab in enumerate(labels):
        result[index][lab] = 1
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)  # axis=1表示按行求和，keeping保证结果和原来形状保持一致，不会少维数
    return ex / sum_ex


if __name__ == '__main__':
    train_datas = load_images("data\\train-images.idx3-ubyte") / 255
    train_label = make_one_hot(load_labels("data\\train-labels.idx1-ubyte"))

    test_datas = load_images("data\\t10k-images.idx3-ubyte") / 255
    test_label = load_labels("data\\t10k-labels.idx1-ubyte")

    epoch = 100
    batch_size = 200
    lr = 0.01

    hidden_num = 256
    w1 = np.random.normal(0, 1, size=(784, hidden_num))
    w2 = np.random.normal(0, 1, size=(hidden_num, 10))

    # 这里不采用Dataset和DataLoader，直接将数据根据batch_size分割
    batch_times = int(np.ceil(len(train_datas) / batch_size))
    for e in range(epoch):
        for batch_index in range(batch_times):
            # 下面为取数据，相当于DataLoader
            batch_x = train_datas[batch_index * batch_size:(batch_index + 1) * batch_size]
            batch_label = train_label[batch_index * batch_size:(batch_index + 1) * batch_size]
            # 推理过程forward
            h = batch_x @ w1
            sig_h = sigmoid(h)
            p = sig_h @ w2
            pre = softmax(p)

            loss = -np.sum(batch_label * np.log(pre))/batch_size  # 多元交叉熵
            # 方向传播backward
            G2 = (pre - batch_label)/batch_size # 减小G2，防止数据溢出
            delta_w2 = sig_h.T @ G2
            delta_sig_h = G2 @ w2.T
            delta_h = delta_sig_h * sig_h * (1 - sig_h)
            delta_w1 = batch_x.T @ delta_h
            # 更新 w1 w2
            w1 = w1 - lr * delta_w1
            w2 = w2 - lr * delta_w2

        print(f"当前第{e+1}轮")
        print(f"当前loss值:{loss}")

        h = test_datas @ w1
        sig_h = sigmoid(h)
        p = sig_h @ w2
        pre = softmax(p)
        pre=np.argmax(pre,axis=1) # 求最大数的下标

        acc=np.sum(pre==test_label)/len(test_label)

        print(f"准确度：{acc}")
