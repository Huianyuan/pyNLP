import random


class MyDataset:
    def __init__(self, all_datas, batch_size, shuffle=True):
        self.all_datas = all_datas
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cursor = 0  # 数据边界

    def __iter__(self):  # python魔术方法：某种场景下自动触发的方法
        if shuffle:
            random.shuffle(self.all_datas)
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self.all_datas):
            raise StopIteration

        batch_data = self.all_datas[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return batch_data


if __name__ == "__main__":
    all_datas = [1, 2, 3, 4, 5, 6, 7]
    batch_size = 2
    shuffle = True
    epoch = 2

    dataset = MyDataset(all_datas, batch_size, shuffle)
    for e in range(epoch):
        for batch_data in dataset:
            print(batch_data)
