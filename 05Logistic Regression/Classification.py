import numpy as np

"""
猫狗分类问题demo
特征为两个，可以表示为7X2的矩阵，猫狗两类，做拼接就是14X2的矩阵，记为X，做K矩阵（2X1），计算预测值pre=X @ K，
然后用 激活函数 将pre值映射到label区间，然后进行类别选择。这里激活函数为sigmoid函数，做二分类较多
p = X @ K , pre = sig(p) , loss = label*log(pre) + (1-label)*log(1-pre) 交叉熵损失

梯度计算： G = ∂loss/∂p = ∂loss/∂pre * ∂pre/∂p = pre-label （常数省略）
由推导公式 A * B = C , G = ∂l/∂C , ▽A = G * B^T ' ▽B = A^T * G 得更新后的K值为：
    ∂loss/∂k = X^T @ G , k = k - lr * ∂loss/∂k
"""


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    # m毛发长，腿长
    dogs = np.array([[8.9, 12], [9, 11], [10, 13], [9.9, 11.2], [12.2, 10.1], [9.8, 13], [8.8, 11.2]],
                    dtype=np.float32)  # 0
    cats = np.array([[3, 4], [5, 6], [3.5, 5.5], [4.5, 5.1], [3.4, 4.1], [4.1, 5.2], [4.4, 4.4]], dtype=np.float32)  # 1

    # 猫狗标签为1和0，两数据量都为7,生成14X1的矩阵
    labels = np.array([0] * 7 + [1] * 7, dtype=np.int32).reshape(-1, 1)

    X = np.vstack((dogs, cats))

    k = np.random.normal(0, 1, size=(2, 1))
    b = 0
    epoch = 1000
    lr = 0.05

    for e in range(epoch):
        p = X @ k + b
        pre = sigmoid(p)
        loss = - np.sum(labels * np.log(pre) + (1 - labels) * np.log(1 - pre))  # 加 - 号是因为0~1之间的log为负数，取反

        G = pre - labels
        delta_k = X.T @ G
        delta_b = np.sum(G)

        k = k - lr * delta_k
        b = b - lr * delta_b

        print(loss)

while True:
    f1 = float(input("请输入毛发长度："))
    f2 = float(input("请输入腿长："))

    test_x = np.array([f1, f2]).reshape(1, 2)
    p = sigmoid(test_x @ k + b)
    if p>0.5: # 猫标识为1，预测值大于0.5即可判断为猫
        print("类别：猫")
    else:
        print("类别：狗")


