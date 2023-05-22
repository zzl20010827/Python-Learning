import numpy as np
import math
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def in_product(x, y):
    """
    内积
    """
    return np.dot(x-y, x-y)


def gauss_ker(x, y, sig):
    """
    gauss核函数
    """
    return math.exp(-in_product(x, y)/(2*sig**2))


def obj(x, y_train, x_train, alph, b, sig):
    """
    目标函数
    """
    kappa = [gauss_ker(xi, x, sig) for xi in x_train]
    return sum(alph*y_train*kappa)+b[0]


def takestep(i1, i2, alph, x_train, y_train, err, b, C, sig):
    eps = 0.001  # 精度
    if (i1 == i2):
        return 0
    y1 = y_train[i1]
    y2 = y_train[i2]
    e1 = obj(x_train[i1], y_train, x_train, alph, b, sig)-y1
    e2 = obj(x_train[i2], y_train, x_train, alph, b, sig)-y2
    s = y1*y2
    if y1 == y2:
        L = max(0, alph[i2]+alph[i1]-C)
        H = min(C, alph[i2]+alph[i1])
    else:
        L = max(0, alph[i2]-alph[i1])
        H = min(C, C+alph[i2]-alph[i1])
    if L == H:
        return 0
    k11 = gauss_ker(x_train[i1], x_train[i1], sig)
    k12 = gauss_ker(x_train[i1], x_train[i2], sig)
    k22 = gauss_ker(x_train[i2], x_train[i2], sig)
    eta = k11+k22-2*k12
    if eta > 0:
        a2 = alph[i2]+y2*(e1-e2)/eta
        if a2 < L:
            a2 = L
        if a2 > H:
            a2 = H
    else:
        temp = alph[i2]
        alph[i2] = L
        Lobj = obj(x_train[i2], y_train, x_train, alph, b, sig)-y2
        alph[i2] = H
        Hobj = obj(x_train[i2], y_train, x_train, alph, b, sig)-y2
        if Lobj < Hobj-eps:
            a2 = L
        elif Lobj > Hobj+eps:
            a2 = H
        else:
            a2 = temp
        alph[i2] = temp
    if abs(a2-alph[i2]) < eps*(a2+alph[i2]+eps):
        return 0
    a1 = alph[i1]+s*(alph[i2]-a2)
    b1 = -e1-y1*(a1-alph[i1])*k11-y2*(a2-alph[i2])*k12+b[0]
    b2 = -e2-y1*(a1-alph[i1])*k12-y2*(a2-alph[i2])*k22+b[0]
    b[0] = (b1+b2)/2
    err[i1] = e1
    err[i2] = e2
    alph[i2] = a2
    alph[i1] = a1
    return 1


def examineExample(i2, x_train, y_train, alph, b, err, C, sig):
    tol = 0.01    # 松弛变量
    num_train = x_train.shape[0]
    y2 = y_train[i2]
    alph2 = alph[i2]
    e2 = obj(x_train[i2], y_train, x_train, alph, b, sig)-y2
    err[i2] = e2
    r2 = e2*y2
    if (1 < -tol and alph2 < C) or (r2 > tol and alph2 > 0):
        not_bound_num = 0
        for a in alph:
            if a > 0 and a < C:
                not_bound_num += 1
        if not_bound_num > 1:
            if e2 > 0:
                i1 = np.argmin(err)
            else:
                i1 = np.argmax(err)
            if takestep(i1, i2, alph, x_train, y_train, err, b, C, sig):
                return 1
        rand_num = np.linspace(0, num_train, num=num_train,
                               endpoint=False, dtype=int)
        random.shuffle(rand_num)
        for i1 in rand_num:
            if alph[i1] != 0 and alph[i1] != C:
                if takestep(i1, i2, alph, x_train, y_train, err, b, C, sig):
                    return 1
        random.shuffle(rand_num)
        for i1 in rand_num:
            if takestep(i1, i2, alph, x_train, y_train, err, b, C, sig):
                return 1
    return 0


cancer = load_breast_cancer()  # 二分类数据集
x = cancer.data[0:500]  # 样本数据
y = cancer.target[0:500]  # 样本标签
x_test = cancer.data[500:]  # 测试集
y_test = cancer.target[500:]
y[y == 0] = -1
dim = x.shape[1]  # 数据维数
num = x.shape[0]  # 数据个数
num_test = x_test.shape[0]  # 测试集大小
tol = 0.01    # 松弛变量
k = 5  # k重交叉验证
num_train = int(math.floor((1-1/k)*num))    # 训练集样本个数
num_valid = num-num_train  # 测试集样本个数
C_series = np.linspace(0.05, 1, num=k)   # 二次规划超参
sig_series = np.linspace(300, 500, num=k)   # 高斯核超参
alph = 0.1*np.ones(num_train)    # Lagrange乘子
weight = np.zeros(dim)  # 权重
b = np.zeros(1)  # 分离面常数项
err = np.zeros(num_train)  # 误差
alph_best = np.zeros(len(alph))  # 最佳系数
b_best = np.zeros(1)
C_best = 0  # 最佳超参
sig_best = 0
score_mem = 0  # 最佳评价指标
err_valid_mem = np.zeros(num_valid)  # 最佳参数误差
x_train_best = np.zeros(num_train)  # 最佳训练集
y_train_best = np.zeros(num_train)
for time_C in range(0, k):
    for time_sig in range(0, k):
        # 交叉验证
        C = C_series[time_C]
        sig = sig_series[time_sig]
        x_train, x_valid, y_train, y_valid = train_test_split(
            x, y, train_size=num_train)
        # SMO
        numchanged = 0
        examineall = 1
        while numchanged > 0 or examineall == 1:
            numchanged = 0
            if examineall:
                for i in range(0, num_train):
                    numchanged += examineExample(i, x_train,
                                                 y_train, alph, b, err, C, sig)
            else:
                for i in range(0, num_train):
                    if alph[i] != C and alph[i] != 0:
                        numchanged += examineExample(i, x_train,
                                                     y_train, alph, b, err, C, sig)
            if examineall == 1:
                examineall = 0
            elif numchanged == 0:
                examineall = 1
        # 评估
        err_valid = np.zeros(num_valid)
        for i in range(0, num_valid):  # 测试集误差
            temp = obj(x_valid[i], y_train, x_train, alph, b, sig)*y_valid[i]
            err_valid[i] = max(1-tol-temp, 0)
        score = sum(err_valid)  # 评价指标
        if time_C == 0 and time_sig == 0:
            # 初始化记忆
            alph_best = alph
            b_best = b
            score_mem = score
            err_valid_mem = err_valid
            C_best = C
            sig_best = sig
            x_train_best = x_train
            y_train_best = y_train
        elif score < score_mem:
            # 如果现模型更优
            alph_best = alph
            b_best = b
            score_mem = score
            err_valid_mem = err_valid
            C_best = C
            sig_best = sig
            x_train_best = x_train
            y_train_best = y_train
# 选取最优超参在整个训练集上训练
C = C_best
sig = sig_best
x_train = x
y_train = y
num_train = x_train.shape[0]
alph = 0.1*np.ones(num_train)  # Lagrange乘子
b = np.zeros(1)  # 常数项
err = np.zeros(num_train)  # 误差
# SMO
numchanged = 0
examineall = 1
while numchanged > 0 or examineall == 1:
    numchanged = 0
    if examineall:
        for i in range(0, num_train):
            numchanged += examineExample(i, x_train,
                                         y_train, alph, b, err, C, sig)
    else:
        for i in range(0, num_train):
            if alph[i] != C and alph[i] != 0:
                numchanged += examineExample(i, x_train,
                                             y_train, alph, b, err, C, sig)
    if examineall == 1:
        examineall = 0
    elif numchanged == 0:
        examineall = 1
# 评估
err_test = np.zeros(num_test)
for i in range(0, num_test):  # 测试集误差
    temp = obj(x_test[i], y_train, x_train, alph, b, sig)*y_test[i]
    err_test[i] = max(1-tol-temp, 0)
score_test = sum(err_test)  # 评价指标
