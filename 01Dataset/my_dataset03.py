import random
import numpy as np


class MyDataset:
    def __init__(self, all_datas, batch_size, shuffle=True):
        self.all_datas = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):  # python魔术方法：某种场景下自动触发的方法
        if shuffle:
            random.shuffle(self.all_datas)
        return DataLoader(self)

    def __len__(self):
        return len(self.all_datas)


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset
        self.indexs = [i for i in range(len(self.dataset))]
        if self.dataset.shuffle == True:
            np.random.shuffle(self.indexs)
        self.cursor = 0  # 数据边界

    def __next__(self):
        if self.cursor >= len(self.dataset.all_datas):
            raise StopIteration
        index = self.indexs[self.cursor:self.cursor + self.dataset.batch_size]
        batch_data = self.dataset.all_datas[index]
        self.cursor += self.dataset.batch_size
        return batch_data


if __name__ == "__main__":
    all_datas = np.array([1, 2, 3, 4, 5, 6, 7])
    batch_size = 2
    shuffle = True
    epoch = 2

    dataset = MyDataset(all_datas, batch_size, shuffle)
    for e in range(epoch):
        for batch_data in dataset:
            print(batch_data)
