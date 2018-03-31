# -*- coding: utf-8 -*-
# @Author: kicc
# @Date:   2018-03-31 18:08:47
# @Last Modified by:   kicc
# @Last Modified time: 2018-03-31 18:14:56


from itertools import combinations, permutations
from collections import Counter

import xlwt
from sklearn.model_selection import ShuffleSplit
import numpy as np

from sklearn import svm, linear_model, cross_validation

import pandas as pd


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = self.transform_pairwise(X, y)
        # print(X_trans,y_trans)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.ravel())

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        The item is given such that items ranked on top have are
        predicted a higher ordering (i.e. 0 means is the last item
        and n_samples would be the item ranked on top).
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.ravel()))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_tests = self.tst_transform_pairwise(X)
        return super(RankSVM, self).predict(X_tests)

    def tst_transform_pairwise(self, X):
        X_new = []
        perm = permutations(range(X.shape[0]), 2)
        # print(X.shape[0])
        for k, (i, j) in enumerate(perm):
            X_new.append(X[i] - X[j])
        return np.asarray(X_new)

    def transform_pairwise(self, X, y):
        """Transforms data into pairs with balanced labels for ranking
        Transforms a n-class ranking problem into a two-class classification
        problem. Subclasses implementing particular strategies for choosing
        pairs should override this method.
        In this method, all pairs are choosen, except for those that have the
        same target value. The output is an array of balanced classes, i.e.
        there are the same number of -1 as +1
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples,) or (n_samples, 2)
            Target labels. If it's a 2D array, the second column represents
            the grouping of samples, i.e., samples with different groups will
            not be considered.
        Returns
        -------
        X_trans : array, shape (k, n_feaures)
            Data as pairs
        y_trans : array, shape (k,)
            Output class labels, where classes have values {-1, +1}
        """
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
            # output balanced classes

        X_res = np.asarray(X_new)
        y_res = np.asarray(y_new)
        return X_res, y_res

    def rank_list(self, y, length):
        '''

        :param y:预测出的 -1 +1 序列 就为[ 1,-1,1,-1,1,-1,-1,1,-1,1,1,-1]
        :param length: 原来测试数据的长度 如测试数据为(x1,x2,x3,x4)，这里length即为4
        :return: 排序后从小到大的下标
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

    def predict2(self, X):
        y_pred = self.score(X)
        length = X.shape[0]
        count = self.rank_list(y_pred, length)
        pred_bug_rank = self.trans(count=count)
        return pred_bug_rank

    def trans(self, count):
        t = []
        for i, j in enumerate(count):
            t.append((i, j))

        t = sorted(t, key=lambda k: k[1])
        res = []

        for i in t:
            res.append(i[0])

        return res
