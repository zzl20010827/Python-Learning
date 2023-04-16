import numpy as np
import random as rand
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import kmeans
from kmeans import dist  # type: ignore
import time

# 设置图像中文字体为黑体
mpl.rc("font", family="SimHei")
# 解决坐标轴负号不显示
mpl.rcParams['axes.unicode_minus'] = False


def sqdLoss(data, mean, k=5, n=100):
    """ 均方损失函数值 """
    loss = 0
    for i in range(0, k):
        for j in range(0, n):
            if data[j, 2] == i:
                loss = loss + dist(data[j, 0:2], mean[i, :])
    return loss


def Metropolis(former, latter, t):
    """ 按照Metropolis准则输出1 """
    x = rand.uniform(0, 1)
    cumulative_probability = 0.0
    prob = math.exp(-(latter - former) / t)
    for item, item_probability in zip([1, 0], [prob, 1 - prob]):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


class impkmeans():
    """ 模拟退火改进k均值算法 """

    def __init__(self):
        """ 生成初始解 """
        # 进行一次k均值迭代生成初始解
        former = kmeans.kmean()  # type: ignore
        former.iteration()
        former.Loss()
        former.kmeanplt()
        self.data = former.data
        self.k = former.k
        self.n = former.n
        self.mean = former.mean
        self.distm = former.distm

    def iteration(self):
        """ 迭代 """
        # 记录初始解
        initdata = self.data
        initmean = self.mean
        initdistm = self.distm
        initloss = sqdLoss(initdata, initmean)
        # 初始温度、退火因子和终止温度
        T = 100
        Ttermi = 0.01
        x = 0.95
        # 扰动点的数量
        change = 50
        # 迭代次数
        iteration = 10
        while T >= Ttermi:
            for iter in range(0, iteration):
                # 随机扰动后的样本和均值向量
                testdata = self.data
                testmean = self.mean
                testdistm = self.distm
                # 随机选取扰动点
                randot = np.random.randint(low=0, high=self.n, size=change)
                # 将扰动点随机分配给其他类
                for i in randot:
                    while True:
                        choice = rand.choice(np.linspace(
                            0, self.k - 1, self.k))
                        if choice != self.data[i, 2]:
                            testdata[i, 2] = choice
                            break
                # 更新均值向量
                for i in range(0, self.k):
                    smean = (0.0, 0.0)
                    num = 0
                    for j in range(0, self.n):
                        if testdata[j, 2] == i:
                            smean = smean + testdata[j, 0:2]
                            num = num + 1
                    if num != 0:
                        testmean[i, :] = [i / num for i in smean]
                # 重新计算距离
                for i in range(0, self.n):
                    for j in range(0, self.k):
                        testdistm[i, j] = dist(testdata[i, 0:2],
                                               testmean[j, :])
                # 计算损失函数
                loss = sqdLoss(self.data, self.mean)
                testloss = sqdLoss(testdata, testmean)
                # 随机扰动的结果优于现结果
                if testloss < loss:
                    self.data = testdata
                    self.mean = testmean
                    self.distm = testdistm
                # 随机扰动结果不优于现结果
                elif Metropolis(loss, testloss, T) == 1:
                    # 按Metropolis准则接受现结果
                    self.data = testdata
                    self.mean = testmean
                    self.distm = testdistm
                    # 重新k均值迭代
                    iiter = 100
                    for t in range(0, iiter):
                        # 计算距离
                        for i in range(0, self.n):
                            for j in range(0, self.k):
                                self.distm[i,
                                           j] = dist(self.data[i, 0:2],
                                                     self.mean[j, :])
                        # 分类
                        for i in range(0, self.n):
                            self.data[i, 2] = np.argmin(self.distm[i, :])
                        # 更新均值向量
                        for i in range(0, self.k):
                            smean = (0.0, 0.0)
                            num = 0
                            for j in range(0, self.n):
                                if self.data[j, 2] == i:
                                    smean = smean + self.data[j, 0:2]
                                    num = num + 1
                            if num != 0:
                                self.mean[i, :] = [i / num for i in smean]
            T = T * x
        # 迭代完成后, 若优于初始解则采用, 否则不采用
        loss = sqdLoss(self.data, self.mean)
        if initloss < loss:
            self.data = initdata
            self.mean = initmean
            self.distm = initdistm

    def Loss(self):
        """ 损失函数值 """
        print("改进损失函数值: ", sqdLoss(self.data, self.mean))

    def impkmeanplt(self):
        """ 数据可视化 """
        colors = ['red', 'green', 'blue', 'purple', 'yellow']
        for i in range(0, self.k):
            x = []
            y = []
            for j in range(0, self.n):
                if self.data[j, 2] == i:
                    x.append(self.data[j, 0])
                    y.append(self.data[j, 1])
            plt.scatter(x, y, c=colors[i], s=10)
            plt.scatter(self.mean[i, 0], self.mean[i, 1], c=colors[i], s=100)
        plt.title("改进k均值分类结果")
        plt.savefig("C:/Users/Zhang/Desktop/改进k均值结果.pdf")
        plt.show()


# 开始计算时间
start = time.time()
test = impkmeans()
test.iteration()
# 结束计算时间
end = time.time()
test.Loss()
print('运行时间: ' + str(end - start) + ' s')
test.impkmeanplt()
