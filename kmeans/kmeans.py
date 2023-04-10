import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置图像中文字体为黑体
mpl.rc("font", family="SimHei")
# 解决坐标轴负号不显示
mpl.rcParams['axes.unicode_minus'] = False


def dist(a, b):
    """ a和b点的距离的平方 """
    return np.dot(a - b, a - b)


class kmean():
    """ k均值算法 """

    def __init__(self, k=5, n=100):
        """ k均值迭代 """
        # 聚类k个数和数据n个数
        self.k = k
        self.n = n
        # 生成的数据矩阵,最后一列是点的标签
        self.data = np.random.randn(self.n, 3) * 10
        for i in range(0, self.n):
            self.data[i, 2] = -1
        # 选取初始向量
        mu = np.random.randint(low=0, high=self.n, size=self.k)
        self.mean = np.full((self.k, 2), 0.0)
        for i in range(0, self.k):
            for j in (0, 1):
                self.mean[i, j] = self.data[mu[i], j]
        # 点到均值向量的距离矩阵
        distm = np.full((self.n, self.k), 0.0)
        # 迭代
        iter = 500
        for t in range(0, iter):
            # 计算距离
            for i in range(0, self.n):
                for j in range(0, self.k):
                    distm[i, j] = dist(self.data[i, 0:2], self.mean[j, :])
            # 分类
            for i in range(0, self.n):
                self.data[i, 2] = np.argmin(distm[i, :])
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

    def Loss(self):
        """ 损失函数值 """
        loss = 0
        for i in range(0, self.k):
            for j in range(0, self.n):
                if self.data[j, 2] == i:
                    loss = loss + dist(self.data[j, 0:2], self.mean[i, :])
        print("原损失函数值: ", loss)

    def kmeanplt(self):
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
        plt.title("原k均值分类结果")
        plt.show()


'''
test = kmean()
test.Loss()
test.kmeanplt()
'''
