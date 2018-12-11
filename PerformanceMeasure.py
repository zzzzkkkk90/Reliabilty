import numpy as np
from sklearn import metrics


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

    def CLC(self):
        '''
        CLC和FPA线性相关，在A Learning-to-Rank Approach to Software Defect Prediction这篇论文有证明
        '''
        K = len(self.real)
        N = np.sum(self.real)
        sort_axis = np.argsort(self.pred)
        testBug = np.array(self.real)
        testBug = testBug[sort_axis]
        P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
        CLC=P-(1.0/(2*K))
        return CLC
    
    
    def PofB20(self, percentage):
        '''
        检测排序前20%的模块，能检测出百分之多少的缺陷,Percentage of Bugs
        有10个模块m1,m2,m3,m4，...,m1o. 真实缺陷个数分别为1,4,2,1,5,1,3,1,6,1,self.real=[1,4,2,1,5,1,3,1,6,1]
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,....,m10缺陷个数为5，self.pred=[0,3,5,1,3,4,7,0,1,1]
        预测出排序在前20%的模块为m7，m3
        pofb20=(2+3)/(1+4+2+1+5+1+3+1+6+1)=0.28
        '''
        tuple_real, tuple_pred = [], []
        for idx, real in enumerate(self.real):
            tuple_real.append((idx + 1, real))

        for idx, pred in enumerate(self.pred):
            tuple_pred.append((idx + 1, pred))

        sorted_real = sorted(tuple_real, key=lambda x: x[1], reverse=True)
        sorted_pred = sorted(tuple_pred, key=lambda x: x[1], reverse=True)

        # print("sorted_real ：", sorted_real)
        # print("sorted_pred ：", sorted_pred)

        length = len(self.real)
        number = int(length * percentage)
        total = 0
        for i in range(number):
            num = sorted_pred[i][0]
            for real in sorted_real:
                if num == real[0]:
                    total += real[1]
                    break

        sum_real = sum(self.real)

        return total / sum_real
    
    
    def OPT(self, codeN):
        '''
        有四个模块m1,m2,m3,m4，真实缺陷个数分别为5，2，1，0,self.real=[5，2，1，0],
        代码行数为10，2，100，50，真实的缺陷密度为[0.5,1,0.01,0]
        预测m1缺陷个数为1，m2缺陷个数为0，m3缺陷个数为1，m4缺陷个数为50,self.pred=[1,0,1,50],
        预测的缺陷密度为[0.1,0,0.01,1]
        optimal model’s curve (0,0),（2/162,0.25）,(12/162,0.875),(112/162,1),(1,1)
        worst model’s curve (0,0),(50/162,0,(150/162,0.125),(160/162,0.75),(1,1)
        prediction model’s curve (0,0),(50/162,0),(60/162,0.625),(160/162,0.75),(1,1)
        from sklearn import metrics
        optimalx = np.array([0,2/162, 12/162, 112/162, 1])
        optimaly = np.array([0,0.25, 0.875, 1, 1])
        optimalauc=metrics.auc(optimalx, optimaly)
        worsetx = np.array([0,50/162, 150/162, 160/162, 1])
        worsety = np.array([0,0.0, 0.125, 0.75, 1])
        worsetauc=metrics.auc(worsetx, worsety)
        predx = np.array([0,50/162, 60/162, 160/162, 1])
        predy = np.array([0,0.0, 0.625, 0.75, 1])
        predauc=metrics.auc(predx, predy)
        popt=1-(optimalauc-predauc)
        minpopt=1-(optimalauc-worsetauc)
        normpopt=(popt-minpopt)/(1-minpopt)
        print (normpopt)
        输出得 0.446265938069
        :param codeN: 代码行数
        :return: Norm(opt)
        '''
        codeN_sum = np.sum(codeN)
        real_sum = np.sum(self.real)

        optimal_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(codeN, self.real)]
        optimal_index = list(np.argsort(optimal_index))
        optimal_index.reverse()

        pred_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(codeN, self.pred)]
        pred_index = list(np.argsort(pred_index))
        pred_index.reverse()

        optimal_X = [0]
        optimal_Y = [0]
        for i in optimal_index:
            optimal_X.append(codeN[i] / codeN_sum + optimal_X[-1])
            optimal_Y.append(self.real[i] / real_sum + optimal_Y[-1])

        pred_X = [0]
        pred_Y = [0]
        for i in pred_index:
            pred_X.append(codeN[i] / codeN_sum + pred_X[-1])
            pred_Y.append(self.real[i] / real_sum + pred_Y[-1])

        optimal_index.reverse()
        mini_X = [0]
        mini_Y = [0]
        for i in optimal_index:
            mini_X.append(codeN[i] / codeN_sum + mini_X[-1])
            mini_Y.append(self.real[i] / real_sum + mini_Y[-1])

        a = [i-j for i,j in zip(optimal_X , [2 / 162, 12 / 162, 112 / 162, 1])]
        b = [i - j for i, j  in zip(optimal_Y , [0.25, 0.875, 1, 1])]

        worse_auc = metrics.auc(list(mini_X), list(mini_Y))
        optimal_auc = metrics.auc(list(optimal_X), list(optimal_Y))
        pred_auc = metrics.auc(list(pred_X), list(pred_Y))
        dotaopt = optimal_auc - pred_auc
        miniopt = 1 - (optimal_auc - worse_auc)
        popt = 1 - dotaopt
        normOPT = (popt - miniopt) / (1 - miniopt)

        return normOPT
    
    #normpopt的另外一种实现方法
    def OPT1(self,codeN):
        codeN_sum = np.sum(codeN)
        pred_sum = np.sum(self.pred)
        real_sum = np.sum(self.real)
        
        if pred_sum == 0:
            pred_sum = 1
            
        if codeN_sum == 0:
            codeN_sum = 1

        optimal_index = [j/i if j!=0 and i!=0 else 0 for i , j in zip(codeN,self.real)]
        optimal_index = list(np.argsort(optimal_index))
        optimal_index.reverse()

        pred_index = [j / i if j!=0 and i!=0 else 0 for i, j in zip(codeN, self.pred)]
        pred_index = list(np.argsort(pred_index))
        pred_index.reverse()

        optimal_X = [0]
        optimal_Y = [0]
        for i in optimal_index:
            optimal_X.append(codeN[i]/codeN_sum + optimal_X[-1])
            optimal_Y.append(self.real[i]/real_sum + optimal_Y[-1])

        optimal_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(optimal_X,optimal_Y):
            if x != prev_x:
                optimal_auc += (x - prev_x) * (y + prev_y)/2.
                prev_x = x
                prev_y = y

        pred_X = [0]
        pred_Y = [0]
        for i in pred_index:
            pred_X.append(codeN[i]/codeN_sum + pred_X[-1])
            pred_Y.append(self.real[i]/real_sum + pred_Y[-1])

        pred_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(pred_X, pred_Y):
            if x != prev_x:
                pred_auc += (x - prev_x) * (y + prev_y)/2.
                prev_x = x
                prev_y = y

        optimal_index.reverse()
        mini_X = [0]
        mini_Y = [0]
        for i in optimal_index:
            mini_X.append(codeN[i] / codeN_sum + mini_X[-1])
            mini_Y.append(self.real[i] / real_sum + mini_Y[-1])
            print("({},{})".format(mini_X[-1], mini_Y[-1]), end="")
        print()

        mini_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(mini_X, mini_Y):
            if x != prev_x:
                mini_auc += (x - prev_x) * (y + prev_y) / 2.
                prev_x = x
                prev_y = y
        mini_auc = 1 - (optimal_auc -mini_auc )
        normOPT = ((1-(optimal_auc - pred_auc)) - mini_auc)/(1 - mini_auc)
        return normOPT
    
    def PofBS20(self, codeN, percentage):
        '''
        检测排序前20%的代码行数，能检测出百分之多少的缺陷,Percentage of Bugs
        有10个模块m1,m1,m2,m3，...,m10. 真实缺陷个数分别为1,4,2,1,5,1,3,1,6,1,self.real=[1,4,2,1,5,1,3,1,6,1]
        代码行数分别为[10,20,30,40,50,60,70,80,90,100]
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,....,m10缺陷个数为5，self.pred=[0,3,5,1,3,4,7,0,1,1]
        预测出缺陷密度排序为m3(0.166) m2(0.149) m7(0.100), m6(0.06), m5(0.059), m4(0.025), m9(0.011), m10(0.01), m1(0.0), m8(0.0)
        pofbs20=(2+4+3)/(1+4+2+1+5+1+3+1+6+1)=0.36
        '''

        preddensity = [j / i for i, j in zip(codeN, self.pred)]

        tuple_real, tuple_pred = [], []
        for idx, real in enumerate(self.real):
            tuple_real.append((idx + 1, real)) #让软件模块的下标是从1开始了，而不是从0开始

        for idx, pred in enumerate(preddensity):
            tuple_pred.append((idx + 1, pred))

        sorted_real = sorted(tuple_real, key=lambda x: x[1], reverse=True)
        sorted_pred = sorted(tuple_pred, key=lambda x: x[1], reverse=True)

        #print("sorted_real ：", sorted_real)
        #print("sorted_pred ：", sorted_pred)
        #print('totalbugs:',sum(self.real))
        #print('totalsloc:',sum(codeN))


        sloc = 0.0
        bugs=0.0
        for i in range(len(self.real)):
            index = sorted_pred[i][0] #预测缺陷密度排在第i位的模块的index
            #print ('预测缺陷密度排在第',i,'位的软件模块的index等于：',index)
            for real in sorted_real:
                if index == real[0]: #如果缺陷密度排在第一位的模块的index
                    bugs += real[1]
                    sloc+=codeN[real[0]]
                    #print('the real number of bugs of this module is',real[1],'the sloc of this module is',codeN[real[0]])
                    #print ('the cumulative bugs is',bugs,',the cumulative sloc is',sloc)
            if sloc>sum(codeN)*percentage:
                break

        return bugs / sum(self.real)

    
    def PofD20(self):
        '''
        检测排序前20%的模块中有百分之多少是真的是有缺陷的, Percentage of defective modules
        有10个模块m1,m2,m3,m4，...,m1o. 真实缺陷个数分别为1,0,0,0,5,0,4,0,0,1,self.real=[1,0,0,0,5,0,1,0,0,1]
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,....,m10缺陷个数为5，self.pred=[0,3,5,1,3,4,7,0,1,1]
        预测出排序在前20%的模块为m7，m3
        pofb20=1/(20%*10)=0.5

        '''
        tuple_real, tuple_pred = [], []
        for idx, real in enumerate(self.real):
            tuple_real.append((idx + 1, real))

        for idx, pred in enumerate(self.pred):
            tuple_pred.append((idx + 1, pred))

        sorted_real = sorted(tuple_real, key=lambda x: x[1], reverse=True)
        sorted_pred = sorted(tuple_pred, key=lambda x: x[1], reverse=True)

        # print("sorted_real ：", sorted_real)
        # print("sorted_pred ：", sorted_pred)

        length = len(self.real)
        number = int(length * 0.2)

        total = 0
        for i in range(number):
            num = sorted_pred[i][0]
            for real in sorted_real:
                if num == real[0]:
                    if real[1] != 0:
                        # 有缺陷
                        total += 1
                    break

        return total / number
    
    
    def ranking(self):
        '''
        检测真实缺陷个数最大的模块被排在了第%位，真实缺陷个数第二大的模块被排到了第几位，.....真实缺陷个数第五大的模块被排到了第几位
        有10个模块m1,m2,m3,m4，...,m10. 真实缺陷个数分别为1,0,3,0,5,0,3,0,0,1,self.real=[1,0,3,0,5,0,3,0,0,1]
        真实的排序为m5>m3>m7>m1>m10>m2>m4>m6>m8>m9
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,....,m10缺陷个数为1，self.pred=[0,3,5,1,3, 4,7,0,1,1]
        预测出来的排序为m7>m3>m6>m2>m5>m4>m9>m10>m1>m8
        检测真实缺陷个数最大的模块m5被排在了第5/10位，真实缺陷个数第二大的模块被排到了第2/10位，
        真实缺陷个数第3大的模块m7被排到了第1/10位，真实缺陷个数第4大的模块m1被排到了第9/10位，真实缺陷个数第5大的模块m10被排到了第8/10 位，
        输出的是个向量[0.5,0.2,0.1,0.9,0.8]
        缺陷个数相同时，模块的排序会不同，比如预测的m1和m8的缺陷个数都是0，有可能m1排在m8前面，有可能后面
        '''
        # 将self.real=[1,0,3,0,5,0,3,0,0,1] ---->  [(1,1), (2,0), (3,3) ... (10, 1)]形式
        tuple_real, tuple_pred = [], []
        for idx, real in enumerate(self.real):
            tuple_real.append((idx + 1, real))

        for idx, pred in enumerate(self.pred):
            tuple_pred.append((idx + 1, pred))

        sorted_real = sorted(tuple_real, key=lambda x: x[1], reverse=True)
        sorted_pred = sorted(tuple_pred, key=lambda x: x[1], reverse=True)
        # print("sorted_real ：", sorted_real)
        # print("sorted_pred ：", sorted_pred)

        num_real = []
        count = 0
        for item in sorted_real:
            num_real.append(item[0])
            count += 1
            if count > 4:
                break

            def helper(num_real):
                """
                检测真实缺陷个数前五大的模块被排到第几位
                param: num_real: like [5, 3, 7, 1, 10]
                根据num_real去sorted_pred中寻找对应的位数
                """
                result_vec = []
                for num in num_real:
                    for idx, pred in enumerate(sorted_pred):
                        # pred = (7 ,7)..
                        if num == pred[0]:
                            result_vec.append(idx + 1)
                            break
                length = len(sorted_pred)
                return [item / length for item in result_vec]

        return helper(num_real=num_real)

 
    def calc_UN(self, type):
        """
        计算csranksvm计算u,n，IRSVM计算u
            U-list = [u_jk, u_jk, u_jk,...,u_10, u_10,...,u_10]
            N-list = [n_jk, n_jk, n_jk,...,n_10, n_10,...,n_10]
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
        elif type=='svm':
            U_list = [1] * len(U_list)
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

    real = np.array([5, 3, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    pred1 = np.array([5, 3, 3, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    pred2 = np.array([5, 3, 3, 0, 1, 1, 0, 1, 0, 0, 0, 0])
    pred3 = np.array([5, 3, 3, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    pred4 = np.array([5, 3, 3, 0, 1, 1, 0, 0, 0, 1, 0, 0])
    pred5 = np.array([5, 3, 3, 0, 1, 1, 0, 0, 0, 0, 1, 0])
    pred6 = np.array([5, 3, 3, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    pred7 = np.array([5, 3, 3, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    pred8 = np.array([5, 3, 3, 1, 0, 1, 0, 1, 0, 0, 0, 0])
    pred9 = np.array([5, 3, 3, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    pred10 = np.array([5, 3, 3, 1, 0, 1, 0, 0, 0, 1, 0, 0])
    pred11 = np.array([5, 3, 3, 1, 0, 1, 0, 0, 0, 0, 1, 0])
    pred12 = np.array([5, 3, 3, 1, 0, 1, 0, 0, 0, 0, 0, 1])


    fpaperfect=PerformanceMeasure(real, real).FPA()
    fpa1=PerformanceMeasure(real, pred1).FPA()
    fpa2=PerformanceMeasure(real, pred2).FPA()
    fpa3=PerformanceMeasure(real,pred3).FPA()
    fpa4 = PerformanceMeasure(real, pred4).FPA()
    fpa5 = PerformanceMeasure(real, pred5).FPA()
    fpa6 = PerformanceMeasure(real, pred6).FPA()
    fpa7 = PerformanceMeasure(real, pred7).FPA()
    fpa8 = PerformanceMeasure(real, pred8).FPA()
    fpa9 = PerformanceMeasure(real, pred9).FPA()
    fpa10 = PerformanceMeasure(real, pred10).FPA()
    fpa11 = PerformanceMeasure(real, pred11).FPA()
    fpa12 = PerformanceMeasure(real, pred12).FPA()
    print (fpaperfect-(fpa1+fpa2+fpa3+fpa4+fpa5+fpa6)/6)


    real = [4, 3, 1, 6, 5, 0, 2, 0, 0, 1]
    pred = [0, 3, 9, 8, 2, 4, 7, 0, 0, 1]
    print(PerformanceMeasure(real, pred).ranking())
    real = [1, 4, 2, 1, 5, 1, 0, 1, 6, 1]
    pred = [0, 3, 5, 1, 3, 4, 7, 0, 1, 1]
    print(PerformanceMeasure(real, pred).PofB20())
    real = [1, 0, 0, 1, 5, 0, 0, 0, 0, 1]
    pred = [0, 3, 5, 9, 3, 4, 7, 0, 1, 1]
    print(PerformanceMeasure(real, pred).PofD20())
    
    real=np.array([1,4,2,1,5,1,3,1,6,1])
    pred=np.array([0,3,5,1,3,4,7,0,1,1])

    codeN=[10,20,30,40,50,60,70,80,90,100]
    print (PerformanceMeasure(real,pred).FPA())
    print(PerformanceMeasure(real, pred).CLC())
    print(PerformanceMeasure(real,pred).PofBS20(codeN,0.2))
