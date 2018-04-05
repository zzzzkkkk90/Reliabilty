from itertools import combinations, permutations
import numpy as np
from sklearn import svm, linear_model, cross_validation
import pandas as pd


class RankSVM(svm.LinearSVC):

    def __init__(self, C=1.0):
        super().__init__(C=C)

    #X,y为原始缺陷训练集中软件模块的特征和缺陷个数
    #输出为模块对
    def transform_pairwise(self, X, y):
        X_new = []
        y_new = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = np.c_[y, np.ones(y.shape[0])]
        comb = combinations(range(X.shape[0]), 2)
        for k, (i, j) in enumerate(comb):
            # if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            if y[i, 0] == y[j, 0]:
                # skip if same target or different group
                continue
            X_new.append(X[i] - X[j])
            y_new.append(np.sign(y[i, 0] - y[j, 0]))  # -1/1

        X_pairs = np.asarray(X_new)
        y_pairs = np.asarray(y_new)
        return X_pairs, y_pairs


    #X为测试集中软件模块的特征
    def tst_transform_pairwise(self, X):
        X_new = []
        perm = permutations(range(X.shape[0]), 2)
        # print(X.shape[0])
        for k, (i, j) in enumerate(perm):
            X_new.append(X[i] - X[j])
        return np.asarray(X_new)


    def fit(self, X, y):
        X_pairs, y_pairs = self.transform_pairwise(X, y)
        # print(X_trans,y_trans)
        super(RankSVM, self).fit(X_pairs, y_pairs)
        return self



    def predict2(self, X):
        X_tests = self.tst_transform_pairwise(X)
        y_pred = super(RankSVM, self).predict(X_tests)

        length = X.shape[0]

        count = self.rank_list(y_pred, length)
        pred_bug_rank = self.trans(count=count)

        return pred_bug_rank


    def rank_list(self, y, length):
        '''
        比如测试集中软件模块为x0,x1,x2,x3，y为[1,-1,1,   -1,-1,1,   1,1,1,   -1,-1,-1]
        则表明x0>x1,x0<x2,x0>x3,   x1<x0,x1<x2,x1<x3  x2>x0,x2>x1,x2>x3   x3<x0,x3<x1,x3<x2
        最终得出x3<x1<x0<x2
        length: 测试集中软件模块个数 这里length即为4
        :return: 排序后从小到大的下标，这里为3,1,0,2
        '''
        # 计数数组
        count_list = []
        # 计算xi -1 的个数
        for i in range(length):
            count = 0
            for j in range(length - 1):
                n = i * (length - 1) + j
                if(y[n] == -1):
                    count = count + 1
            count_list.append(count)

        k = 0
        max_list = []
        # 选出最小的，处理相同个数的 -1 ，并且把已处理的位置标记为 -1
        while(k < len(count_list)):
            k = k + 1
            large = max(count_list)
            max_index = [m for m in range(length) if count_list[m] == large]
            if len(max_index) > 1:
                for i in range(len(max_index) - 1):
                    max_i = max_index[i]
                    n = max_i * (length - 1) + max_index[i + 1] - 1
                    if(y[n] == 1):
                        max_i = max_index[i + 1]
                max_list.append(max_i)
                count_list[max_i] = -1
            else:
                max_list.append(max_index[0])
                count_list[max_index[0]] = -1
        # 返回的就是一个从小到大的下标
        return max_list

    #对得到的从小到大的模块排序，赋予每个模块不同的缺陷数目
    #比如预测模块的缺陷个数排序为x3<x1<x0<x2，则y0=2，y1=1，y2=3，y3=0，返回值为[2,1,3,0]
    def trans(self, count):
        t = []
        for i, j in enumerate(count):
            t.append((i, j))

        t = sorted(t, key=lambda k: k[1])
        res = []

        for i in t:
            res.append(i[0])

        return res

'''
if __name__ == '__main__':
    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6],[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]])
    y = np.array([5, 4, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

    X_trans = RankSVM().rank_list([1,-1,1,-1,-1,1,1,1,1,-1,-1,-1],4)
    x_count=RankSVM().trans(X_trans)
    print(X_trans)
    print(x_count)
'''
