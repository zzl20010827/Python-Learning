import numpy
import matplotlib.pyplot as plt


def dist(a, b):
    """ a和b点的距离 """
    return numpy.dot(a - b, a - b)


# 聚类k个数和数据n个数
k = 5
n = 100
# 生成的数据矩阵,最后一列是点的标签
data = numpy.random.randn(n, 3) * 10
for i in range(0, n):
    data[i, 2] = -1
# 选取初始向量
mu = numpy.random.randint(low=0, high=n, size=k)
mean = numpy.full((k, 2), 0.0)
for i in range(0, k):
    for j in (0, 1):
        mean[i, j] = data[mu[i], j]
# 点到均值向量的距离矩阵
distm = numpy.full((n, k), 0.0)
# 迭代
iter = 500
for t in range(0, iter):
    # 计算距离
    for i in range(0, n):
        for j in range(0, k):
            distm[i, j] = dist(data[i, 0:2], mean[j, :])
    # 分类
    for i in range(0, n):
        data[i, 2] = numpy.argmin(distm[i, :])
    # 更新均值向量
    for i in range(0, k):
        smean = (0.0, 0.0)
        num = 0
        for j in range(0, n):
            if data[j, 2] == i:
                smean = smean + data[j, 0:2]
                num = num + 1
        if num != 0:
            mean[i, :] = [i / num for i in smean]
# 数据可视化
colors = ['red', 'green', 'blue', 'purple', 'yellow']
for i in range(0, k):
    x = []
    y = []
    for j in range(0, n):
        if data[j, 2] == i:
            x.append(data[j, 0])
            y.append(data[j, 1])
    plt.scatter(x, y, c=colors[i], s=10)
    plt.scatter(mean[i, 0], mean[i, 1], c=colors[i], s=100)
plt.show()
