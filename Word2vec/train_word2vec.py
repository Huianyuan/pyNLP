import numpy as np
import pandas as pd
import pickle
import jieba
import os
from tqdm import tqdm  # 查看循环时间


# skip_gram模型 当前词预测其他值
def load_stop_words(file="stopwords.txt"):  # 停用词加载
    with open(file, "r", encoding="utf-8") as f:
        return f.read().split("\n")  # 返回list格式


def cut_words(file="数学原始数据.csv"):
    stop_words = load_stop_words()  # 加载停用词
    result = []  # 切词结果
    all_data = pd.read_csv(file, encoding="gbk", names=["data"])["data"]
    for words in all_data:
        c_words = jieba.lcut(words)  # 切词，对list
        # 列表推导式，如果当前次不在停用词里则加入切词结果
        result.append([word for word in c_words if word not in stop_words])
    return result


# 计算三大参数：Word_2_index,index_2_word,word_2_onehot
def get_dict(data):
    index_2_word = []
    # words 是每一段的分词结果，即很多词语，列表存储；word则是列表中的每个元素,即词语
    for words in data:
        for word in words:
            if word not in index_2_word:
                index_2_word.append(word)

    word_2_index = {word: index for index, word in enumerate(index_2_word)}
    word_size = len(word_2_index)  # 整个词库的规模

    word_2_onehot = {}
    for word, index in word_2_index.items():
        one_hot = np.zeros((1, word_size))  # 初始化one_hot编码，为1×word_size的向量
        one_hot[0, index] = 1  # 为每一个词进行onehot编码
        word_2_onehot[word] = one_hot  # 将编码后的值添加进word_2_onehot字典
    return word_2_index, index_2_word, word_2_onehot


def softmax(x):
    ex = np.exp(x)
    # keepding参数保证sum结果还是矩阵，而不是向量
    return ex / np.sum(ex, axis=1, keepdims=True)


if __name__ == "__main__":

    data = cut_words()
    word_2_index, index_2_word, word_2_onehot = get_dict(data)

    word_size = len(word_2_index)  # 词库大小
    embedding_num = 107  # 词向量维度
    lr = 0.01  # 学习率
    epoch = 10  # 训练轮次
    n_gram = 3  # 当前词的上下文范围，即相关词范围

    w1 = np.random.normal(-1, 1, size=(word_size, embedding_num))  # 从正态（高斯）分布中抽取随机样本。
    w2 = np.random.normal(-1, 1, size=(embedding_num, word_size))

    for e in range(epoch):  # 轮次
        for words in tqdm(data):  # 每一段(即每篇文章)
            for n_index, now_word in enumerate(words):  # 当前词进行计算,n_index当前词语的索引位置
                now_word_onehot = word_2_onehot[now_word]  # 获取当前词的onehot值
                # 其他词的计算，范围。不包含当前词，其前n_gram个和后n_gram个，前得考虑数据越界，后则切片自动处理
                other_words = words[max(n_index - n_gram, 0):n_index] + words[n_index + 1:n_index + 1 + n_gram]
                for other_word in other_words:
                    other_word_onehot = word_2_onehot[other_word]  # 相关词的onehot

                    hidden = now_word_onehot @ w1  # 隐层
                    p = hidden @ w2
                    pre = softmax(p)  # 预测值

                    # loss = -np.sum(other_word_onehot * np.log(pre)) # 其他词作为真值（标签），匹配时会有无意义短语

                    # 模型梯度更新过程
                    # 矩阵求导公式：A @ B = C，已知delta_C = G，则delta_A =G @ B.t ,delta_B = A.t @ G
                    # 这过程有点迷，没看懂。待查资料学习
                    G2 = pre - other_word_onehot  # 梯度
                    delta_w2 = hidden.T @ G2
                    G1 = G2 @ w2.T
                    delta_w1 = now_word_onehot.T @ G1

                    w1 -= lr * delta_w1
                    w2 -= lr * delta_w2

    with open("word2vec.pkl", "wb") as f:
        pickle.dump([w1, word_2_index, index_2_word], f) # 运算很慢，负采样优化。待学习
