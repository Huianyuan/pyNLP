import random

list1 = [1, 2, 3, 4, 5, 6, 7] # 所有数据量

batch_size = 2  # 一次进行数据处理的量，这里两个两个取
epoch = 2  # 训练轮次
shuffle = True  # 是否将序列的所有元素随机排序。

for e in range(epoch):
    if shuffle:
        random.shuffle(list1)
    for i in range(0, len(list1), batch_size):
        batch_data = list1[i:i + batch_size]
        print(batch_data)
