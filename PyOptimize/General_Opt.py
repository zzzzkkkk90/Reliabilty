class Optimizer(object):
    """docstring for Optimizer"""

    def __init__(self, arg):
        super(Optimizer, self).__init__()
        self.arg = arg


class Test_function(object):
    """docstring for Test_function"""

    def __init__(self):
        super(Test_function, self).__init__()

    def Rosenbrock(self, x):
        """
        高阶Rosenbrock函数
        给出最优解
            最优解仍然应在所有x为1时
        :param x:
        :return:
        """
        function_sum = 0.0
        for i in range(len(x) - 1):
            function_sum += (1 - x[i]) ** 2 + 100 * \
                ((x[i + 1] - x[i] ** 2) ** 2)
        f = function_sum
        return f

    def Loss(self, x, P, r, u, n):
        """
        20维w求解

        """
        def ed_dist(w):
            sqrt_sum = 0.0
            for i, wi in enumerate(w):
                sqrt_sum += wi**2
            return sqrt_sum

        def dot(w, Pi):
            wp_sum = 0.0
            for i in range(len(w)):
                wp_sum += w[i] * Pi[i]
            return float(wp_sum)

        function_sum = 0.0
        for i in range(len(P)):
            function_sum += u[i] * n[i] * max((1 - r[i] * dot(x, P[i])), 0)
        f = function_sum + 0.5 * ed_dist(x)
        return f
    
    def LTR(self, a, X, y):
        """
        ListWise中ga的应用
        param: a是权重向量
        param: X, y 训练集的数据，可以是降序过的
        流程：每次迭代得到一个x向量，与传入的X进行操作得到pred_y,
             然后与真实的y进行fpa值的计算。
        return: max-FPA。
        """

        def FPA(real, pred):
            '''
            有四个模块m1,m2,m3,m4，真实缺陷个数分别为1，4，2，1,self.real=[1，4，2，1]
            预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,self.pred=[0,3,5,1]
            预测出的排序为m3>m2>m4>m1
            fpa=1/4 *1/8 *(4*2+3*4+2*1+1*1)=0.718
            '''
            K = len(real)
            N = np.sum(real)
            sort_axis = np.argsort(pred)
            testBug = np.array(real)
            testBug = testBug[sort_axis]
            P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
            return float(P)

        pred_y = []
        for xi in X:
            yi = np.dot(a, xi)
            yi = float(yi)
            pred_y.append(yi)

        function_fpa = FPA(y, pred_y)
