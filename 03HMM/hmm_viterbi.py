# coding: utf-8
import pickle
from tqdm import tqdm
import numpy as np
import os


def make_label(text_str):  # 为词语编码，今天——》BE  火龙果——》BME  水——》S
    text_len = len(text_str)  # 获取长度
    if text_len == 1:  # 单字成词，返回“S”
        return "S"
    return "B" + "M" * (text_len - 2) + "E"  # 除开开头B结尾E，剩下的都是M


def text_to_state(file="all_train_text.txt"):  # 将原始语料库转换为对应的状态文件
    if os.path.exists("all_train_state.txt"):  # 存在文件则退出
        return
    all_data = open(file, "r", encoding="utf-8").read().split("\n")  # 打开文件并按行切分到 all_data 中（list）
    with open("all_train_state.txt", "w", encoding="utf-8") as f:  # 打开待写入的文件
        for d_index, data in tqdm(enumerate(all_data)):  # 逐行遍历，tqdm为进度条提示，data是一篇文章，有可能为空
            if data:  # data不为空
                state_ = ""
                for w in data.split(" "):  # 当前文章按照空格切分，w是文章中的一个词语
                    if w:  # 词不为空，则调用make_label 方法，对词进行编码转换，并将结果放入state_ 中
                        state_ = state_ + make_label(w) + " "
                    if d_index != len(all_data) - 1:  # 最后一行不要加“\n” 其他行都加“\n”
                        state_ = state_.strip() + "\n"  # 每一行都去掉最后的空格
                    f.write(state_)  # 写入文件，state_是一个字符串


# 定义HMM类，计算三大矩阵
class HMM:
    def __init__(self, file_text="all_train_text.txt", file_state="all_train_state.txt"):
        self.all_states = open(file_state, "r", encoding="utf-8").read().split("\n")[:200]  # 按行获取所有状态
        self.all_texts = open(file_text, "r", encoding="utf-8").read().split("\n")[:200]  # 按行获取所有文本
        self.states_to_index = {"B": 0, "M": 1, "S": 2, "E": 3}  # 给每个状态定义一个索引，以后可以根据状态获取索引
        self.index_to_states = ["B", "M", "S", "E"]  # 根据索引获取状态
        self.len_states = len(self.states_to_index)  # 状态长度：初始为4

        self.init_matrix = np.zeros((self.len_states))  # 初始矩阵；1X4 对应BMSE
        self.transfer_matrix = np.zeros((self.len_states, self.len_states))  # 转移状态矩阵：4x4
        # 发射矩阵，使用二级字典嵌套，初始化了total键，存储当前状态出现的总次数，为了后面的归一化使用
        self.emit_matrix = {"B": {"total": 0}, "M": {"total": 0}, "S": {"total": 0}, "E": {"total": 0}}

    # 计算 初始矩阵
    def cal_init_matrix(self, state):
        self.init_matrix[self.states_to_index[state[0]]] += 1  # BMSE四种状态，对应状态出现1次就+1

    # 计算 转移矩阵
    def cal_transfer_matrix(self, states):
        sta_join = "".join(states)  # 状态转移 从当前状态转移到后一状态，即从sta1每一元素转移到sta2中
        sta1 = sta_join[:-1]
        sta2 = sta_join[1:]
        for s1, s2 in zip(sta1, sta2):  # 同时遍历s1，s2
            self.transfer_matrix[self.states_to_index[s1], self.states_to_index[s2]] += 1

    # 计算 发射矩阵
    def cal_emit_matrix(self, words, states):
        for word, state in zip("".join(words), "".join(states)):  # 先把words和states拼接起来再遍历，因为中间有空格
            self.emit_matrix[state][word] = self.emit_matrix[state].get(word, 0) + 1
            self.emit_matrix[state]["total"] += 1  # 注意这里多添加了一个total键，存储当前状态出现的总次数，为了后面的归一化使用

    # 将矩阵归一化
    def normalize(self):
        self.init_matrix = self.init_matrix / np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix / np.sum(self.transfer_matrix, axis=1, keepdims=True)
        self.emit_matrix = {
            state: {word: t / word_times["total"] * 1000 for word, t in word_times.items() if word != "total"}
            for state, word_times in self.emit_matrix.items()}

    # 训练开始，求三个矩阵的过程
    def train(self):
        if os.path.exists("three_matrix.pkl"):  # 如果已经存在训练好的参数，就不训练了
            self.init_matrix, self.transfer_matrix, self.emit_matrix = pickle.load(open("three_matrix.pkl", "rb"))
            return
        for words, states in tqdm(zip(self.all_texts, self.all_states)):  # 按行读取文件，调用3个矩阵求解函数
            words = words.split(" ")  # 原始文件按照空格切分的
            states = states.split(" ")
            self.cal_init_matrix(states[0])  # 计算初始矩阵
            self.cal_transfer_matrix(states)  # 计算转移矩阵
            self.cal_emit_matrix(words, states)  # 计算发射矩阵
        self.normalize()  # 矩阵求解完毕后进行归一化
        pickle.dump(
            [self.init_matrix, self.transfer_matrix, self.emit_matrix], open("three_matrix.pkl", "wb"))  # 保存参数


def viterbi_t(text, hmm):
    states = hmm.index_to_states
    start_p = hmm.init_matrix
    trans_p = hmm.transfer_matrix
    emit_p = hmm.emit_matrix
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[hmm.states_to_index[y]] * emit_p[y].get(text[0], 0)
        path[y] = [y]
    for t in range(1, len(text)):
        V.append({})
        new_path = {}

        # 检测训练的发射矩阵中是否有该字
        neverSeen = text[t] not in emit_p['S'].keys() and text[t] not in emit_p['M'].keys() and text[t] not in emit_p['E'].keys() and text[t] not in emit_p['B'].keys()
        for y in states:
            emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1, 0  # 设置未知字单独成词
            temp = []
            for y0 in states:
                if V[t - 1][y0] > 0:
                    temp.append((V[t - 1][y0] * trans_p[hmm.states_to_index[y0], hmm.states_to_index[y]] * emitP, y0))
            (prob, state) = max(temp)
            V[t][y] = prob
            new_path[y] = path[state] + [y]
        path = new_path

    (prob, state) = max([(V[len(text) - 1][y], y) for y in states])  # 求最大概念的路径

    result = " "  # 拼接结果
    for t, s in zip(text, path[state]):
        result += t
        if s == "S" or s == "E":  # 如果结果是S或者E，说明是单独成词或单词结尾，则添加空格
            result += " "
    return result


if __name__ == "__main__":
    text_to_state()
    text = "虽然一路上队伍里肃静无声"

    hmm = HMM()
    hmm.train()
    result = viterbi_t(text, hmm)

    print(result)
