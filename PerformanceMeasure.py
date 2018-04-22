import numpy as np


class PerformanceMeasure():

    def __init__(self, real_list, pred_list=None):
        self.real = real_list
        self.pred = pred_list
        self.aae_value = []
        self.fpa_value = 0

    def AAE(self):
        '''
        求每一类模块上的平均绝对误差（average absolute error）
        real_list指测试集中每个模块的真实缺陷个数
        pred_list指训练出的回归模型对测试集中每个模块进行预测得出的预测值
        如real_list=[2,3,0,0,1,1,0,5,3]
         pred_list=[1,1,1,0,1,0,0,3,4]
         输出结果就为0:0.33, 1:0.5,  2:1,  3:1.5,  5:2
        '''
        only_r = np.array(list(set(self.real)))
        # only_r=[0,1,2,3,5]

        for i in only_r:
            r_index = np.where(self.real == i)
            # i=0的时候，r_index=(array([2, 3, 6]), ) 得到是一个tuple

            sum = 0

            # i=0的时候，k = [2, 3, 6]
            k = r_index[0]
            sum = abs(self.real[k] - self.pred[k]).sum()
            avg = sum * 1.0 / len(k)
            self.aae_value.append(avg)

        # 直接返回字典
        aae_result = dict(zip(only_r, self.aae_value))
        return aae_result

    def FPA(self):
        '''
        有四个模块m1,m2,m3,m4，真实缺陷个数分别为1，4，2，1,self.real=[1，4，2，1]
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,self.pred=[0,3,5,1]
        预测出的排序为m3>m2>m4>m1
        fpa=1/4 *1/8 *(4*2+3*4+2*1+1*1)=0.718
        '''
        K = len(self.real)
        N = np.sum(self.real)
        sort_axis = np.argsort(self.pred)
        testBug = np.array(self.real)
        testBug = testBug[sort_axis]
        P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
        return P

    def calc_UN(self, type):
        """
        计算cost-sensitive / information retrieval
            U-list = [u_jk, u_jk, u_jk,...,u_10, u_10,...,u_10]
            N-list = [n_jk, n_jk, n_jk,...,n_10, n_10,...,n_10]

        type: 选择是cs 还是 ir
        ir: N-list = [1. 1. 1. ... 1.]
        """
        # y算是train_y
        # y = np.array([0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        #               0, 3, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3])

        from collections import Counter
        # y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 5])

        y = self.real
        # 将y降序排列
        # y = sorted(y, reverse=True)  # 传入的y已经是降序过的

        set_y = set(y)
        set_y = sorted(set_y, reverse=True)
        Counter_y = Counter(y)

        # index_list = [0,1,4,10]
        index_list = []
        index = 0
        for i in set_y:
            index_list.append(index)
            index += Counter_y[i]
        index_list.append(len(y))

        fpa_perfect = PerformanceMeasure(y, y).FPA()
        # print('Perfect fpa =', fpa_perfect)

        index_list_len = len(index_list)
        U_list = []
        N_list = []

        for i in range(index_list_len - 1):
            for j in range(i + 1, index_list_len - 1):

                u, n = self.calc_U_N(origin_y=y, large_list=y[index_list[i]:index_list[i + 1]], index1=index_list[i],
                                     small_list=y[index_list[j]:index_list[j + 1]], index2=index_list[j], fpa_perfect=fpa_perfect)
                U_list += u
                N_list += n
        # print(U_list)
        max_n = max(N_list)
        N_list = [max_n / n for n in N_list]
        # print(N_list)

        if type == 'cs':
            return U_list, N_list
        elif type == 'ir':
            N_list = [1] * len(N_list)
            return U_list, N_list

    def calc_U_N(self, origin_y, large_list, index1, small_list, index2, fpa_perfect):
        """
        将原本的y分割成多份，传入的是两份
        比如y中有3,2,1,0， large_list = [3,3,3,3], small_list = [2,2,2,2,2]
        或者large_list = [2,2,2,2,2], small_list = [1,1,1,1,1,1]
        """
        from copy import deepcopy
        copy_y = deepcopy(origin_y)

        long_len = len(large_list)
        short_len = len(small_list)
        m_jk = long_len * short_len

        sum_fpa = 0.0

        for i, large_num in enumerate(large_list):
            pos_i = i + index1
            for j, small_num in enumerate(small_list):
                pos_j = j + index2
                origin_y[pos_i], origin_y[pos_j] = origin_y[pos_j], origin_y[pos_i]
                fpa = PerformanceMeasure(copy_y, origin_y).FPA()
                origin_y[pos_i], origin_y[pos_j] = origin_y[pos_j], origin_y[pos_i]

                sum_fpa += fpa
        u_jk = fpa_perfect - (sum_fpa) / m_jk
        u_jk = float(u_jk)

        return [u_jk] * m_jk, [m_jk] * m_jk


if __name__ == '__main__':
    # real=np.array([2,3,0,0,1,1,0,5,3])
    # pred=np.array([1,5,1,0,1,0,0,7,2])
    # aeeresult=PerformanceMeasure(real,pred).AEE()
    # print (aeeresult)

    real = np.array([5, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred1 = np.array([1, 5, 1, 1, 0, 0, 0, 0, 0, 0])
    pred2 = np.array([1, 1, 5, 1, 0, 0, 0, 0, 0, 0])
    pred3 = np.array([1, 1, 1, 5, 0, 0, 0, 0, 0, 0])
    print (PerformanceMeasure(real, pred1).FPA())
    print (PerformanceMeasure(real, real).FPA())
    print (PerformanceMeasure(real, pred2).FPA())
    print (PerformanceMeasure(real, pred3).FPA())
