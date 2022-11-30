import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Transformer_Encoder import TransformerEncoder


# 数据读取。num参数控制执行文本的数量，前期可以使用少量文本进行测试
def read_data(train_or_test, num=None):
    with open(os.path.join("../data", train_or_test + ".txt"), encoding="utf-8") as f:
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
def built_curpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}  # PAD（padding）填充词，UNK未收录词
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    # return word_2_index, np.eye(len(word_2_index), dtype=np.float32)  # 返回语料库和np生成的对角矩阵
    # 均值为0，方差为1，大小为xx的矩阵
    # return word_2_index, np.random.normal(0, 1, (len(word_2_index), embedding_num)).astype(np.float32)
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)  # 采用内置方法，需修改57行[]为(),并修改类型


class TextDataset(Dataset):
    def __init__(self, texts, labels, word_2_index, index_2_embedding, max_len):
        self.texts = texts  # 文本
        self.labels = labels  # 标签
        self.word_2_index = word_2_index  # 构造的语料库，即每个字建立索引
        self.index_2_embedding = index_2_embedding  # 索引矩阵
        self.max_len = max_len  # 句子最大长度

    def __getitem__(self, index):
        # 1.根据index获取数据
        text = self.texts[index]
        # label = self.labels[index]
        label = int(self.labels[index])

        text_len = len(text)  # 记录句子原始长度
        # 2.剪裁文本长度至max_len
        text = text[:self.max_len]
        # 3.将 中文文本根据词表，用数字表示
        word_index = [self.word_2_index.get(i, 1) for i in text]  # 中文文本——》index,获取不到时，填充1，代表UNK
        word_index = word_index + [0] * (self.max_len - len(text))  # 填充，0表示PAD
        # 获取文本的表示矩阵，即每个字用多维度表示
        text_embedding = self.index_2_embedding(torch.tensor(word_index))

        return text_embedding, label, text_len

    def __len__(self):
        return len(self.labels)


def test_file():
    global model, device, word_2_index, index_2_embedding, max_len

    test_texts, test_labels = read_data("test")

    test_dataset = TextDataset(test_texts, test_labels, word_2_index, index_2_embedding, max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    result = []
    for text, label in test_dataloader:
        text = text.to(device)
        model(text)
        result.extend(model.pre)
    with open(os.path.join("../data", "test_result_RNN.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join([str(i) for i in result]))

    test_acc = sum([i == int(j) for i, j in zip(result, test_labels)]) / len(test_labels)
    print(f"test acc={test_acc * 100:.2f}%")
    print("test over")


class TransformerClassifier(nn.Module):
    def __init__(self, embedding_num, class_num, device="cup"):
        super().__init__()
        self.transformer = TransformerEncoder(device, embedding_num=embedding_num)
        self.classifier = nn.Linear(embedding_num, class_num)
        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, batch_embedding, batch_len, batch_label=None):
        out = self.transformer(batch_embedding, batch_len)
        out = out[:, 0, :]  # 降维 2X4X200 ，只取每个的第一个字的特征，即2X1X200 2X200

        pre = self.classifier(out)

        if batch_label is not None:
            loss = self.loss_fun(pre, batch_label)
            return loss


if __name__ == '__main__':
    train_texts, train_labels = read_data("train", 200)  # 训练,读数据测试，只读200条
    dev_texts, dev_labels = read_data("dev", 30)  # 验证

    # 断言，判断训练集和测试集中文本与标签数量一致
    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    embedding_num = 100
    max_len = 30
    batch_size = 32
    epoch = 5
    lr = 0.001
    class_num = len(set(train_labels))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index, index_2_embedding = built_curpus(train_texts, embedding_num)

    train_dataset = TextDataset(train_texts, train_labels, word_2_index, index_2_embedding, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    dev_dataset = TextDataset(dev_texts, dev_labels, word_2_index, index_2_embedding, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size=2, shuffle=False)

    model = TransformerClassifier(embedding_num, class_num, device)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epoch):
        for batch_embedding, batch_label, batch_len in train_dataloader:
            batch_embedding = batch_embedding.to(device)
            batch_label = batch_label.to(device)
            loss = model(batch_embedding, batch_len, batch_label)

            loss.backward()
            optim.step()
            optim.zero_grad()

        print(f"loss:{loss:.2f}")
