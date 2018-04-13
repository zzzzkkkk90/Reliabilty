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
