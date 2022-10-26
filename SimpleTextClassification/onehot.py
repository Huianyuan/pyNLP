import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


# 数据读取。num参数控制执行文本的数量，前期可以使用少量文本进行测试
def read_data(train_or_test, num=None):
    with open(os.path.join("data", train_or_test + ".txt"), encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:  # 文本最后一行为空
            t, l = data.split("\t")
            texts.append(t)
            labels.append(l)
    if num == None:
        return texts, labels
    else:
        return texts[:num], labels[:num]


# 构建语料库,word_2_index字典包括训练文本所有的字符
def built_curpus(train_texts):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}  # PAD（padding）填充词，UNK未收录词
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index, np.eye(len(word_2_index), dtype=np.float32)  # 返回语料库和np生成的对角矩阵


class OhDataset(Dataset):
    def __init__(self, texts, labels, word_2_index, index_2_onehot, max_len):
        self.texts = texts  # 文本
        self.labels = labels  # 标签
        self.word_2_index = word_2_index  # 构造的语料库，即每个字建立索引
        self.index_2_onehot = index_2_onehot  # 索引矩阵
        self.max_len = max_len  # 句子最大长度

    def __getitem__(self, index):
        # 1.根据index获取数据
        text = self.texts[index]
        label = int(self.labels[index])
        # 2.剪裁文本长度至max_len
        text = text[:self.max_len]
        # 3.将 中文文本——》index——》onehot 形式，长度不够则填充
        # text_index = [word_2_index[i] for i in text]  # 中文文本——》index
        text_index = [word_2_index.get(i, 1) for i in text]  # 中文文本——》index,获取不到时，填充1，代表UNK
        text_index = text_index + [0] * (self.max_len - len(text_index))  # 填充
        text_onehot = self.index_2_onehot[text_index]  # 获取文本的onehot矩阵

        return text_onehot, label

    def __len__(self):
        return len(self.labels)


class OhModel(nn.Module):
    # 词库大小（19），隐藏层大小（30），分类类别（3），句子最大长度（6）
    def __init__(self, curpus_len, hidden_num, class_num, max_len):
        super().__init__()
        self.linear1 = nn.Linear(curpus_len, hidden_num)  # linear层1
        self.active = nn.ReLU()  # 激活函数
        self.flatten = nn.Flatten()  # Flatten层用来将输入“压平”,即把多维的输入一维化
        self.linear2 = nn.Linear(max_len * hidden_num, class_num)  # linear层2
        self.cross_loss = nn.CrossEntropyLoss()  # 交叉熵损失

    def forward(self, text_onehot, labels=None):
        hidden = self.linear1.forward(text_onehot)
        hidden_act = self.active(hidden)
        hidden_f = self.flatten(hidden_act)
        p = self.linear2(hidden_f)

        # 转换求得预测值，后续进行验证
        self.pre = torch.argmax(p, dim=-1).detach().cpu().numpy().tolist()

        if labels is not None:
            loss = self.cross_loss(p, labels)
            return loss


def test_file():
    global model, device, word_2_index, index_2_onehot, max_len

    test_texts, test_labels = read_data("test")

    test_dataset = OhDataset(test_texts, test_labels, word_2_index, index_2_onehot, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    result = []
    for text, label in test_dataloader:
        text = text.to(device)
        model(text)
        result.extend(model.pre)
    with open(os.path.join("data", "test_result.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([str(i) for i in result]))

    test_acc = sum([i == int(j) for i, j in zip(result, test_labels)]) / len(test_labels)
    print(f"test acc={test_acc*100:.2f}%")
    print("test over")


if __name__ == '__main__':
    train_texts, train_labels = read_data("train", 20000)  # 训练
    dev_texts, dev_labels = read_data("dev")  # 验证

    # 断言，判断训练集和测试集中文本与标签数量一致
    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    epoch = 5  # 训练轮数
    batch_size = 30  # 一次取多少
    max_len = 25  # 句子的最大长度
    hidden_num = 30  # 隐藏层大小
    class_num = len(set(train_labels))  # 类别，即标签数量
    lr = 0.0006  # 学习率

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index, index_2_onehot = built_curpus(train_texts)
    # 训练集
    train_dataset = OhDataset(train_texts, train_labels, word_2_index, index_2_onehot, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    # 验证集
    dev_dataset = OhDataset(dev_texts, dev_labels, word_2_index, index_2_onehot, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = OhModel(len(word_2_index), hidden_num, class_num, max_len)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epoch):
        for texts, labels in tqdm(train_dataloader):
            texts = texts.to(device)
            labels = labels.to(device)

            loss = model(texts, labels)
            loss.backward()

            optim.step()
            optim.zero_grad()

        right_num = 0  # 统计预测正确数量
        for texts, labels in dev_dataloader:
            texts = texts.to(device)

            model(texts)
            # right_num += sum(labels.numpy().tolist() == model.pre)
            right_num += int(sum([i == j for i, j in zip(model.pre, labels)]))

        print(f"dev acc:{right_num / len(dev_labels) * 100:.2f}%")

        # print(f"loss:{loss:.2f}")
