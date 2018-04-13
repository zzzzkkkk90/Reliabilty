import numpy as np

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing

from rankSVM import RankSVM

from myGAFT import pyGaft

import importlib

def bootstrap():

    #dataset = Processing().import_data()
    count = 0
    for dataset, filename in Processing().import_single_data():
        count+=1
        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)

        # print('train shape', training_data_X.shape)
        # 1.降序排列训练集（Processing中已完成）

        # 2.利用transfrom_pairwise() 得到Pi，ri
        # P是一个矩阵，每个向量是两个x相减的结果
        # r是一个向量 因为排序过，所以结果r = [1,1,1,1,1,1...]
        rs = RankSVM()
        P, r = rs.transform_pairwise(training_data_X, training_data_y)
        #print('p shape ', P.shape, 'r len ', len(r))
        P = P.tolist()
        r = r.tolist()
        print('type of P ', type(P[0][0]), 'type of r ', type(r[0]))
        # P = [[1, 1, 2], [1, -1, 3], [3, 2, 1], [1, -5, 1], [2, 1, -2]]
        # r = [1, 1, 1, 1, 1]

        # 3.用training_data_y计算u,n
        u, n = PerformanceMeasure(training_data_y).calc_UN()
        # print(len(u), len(n))
        print(type(u[0]), type(n[0]))

        # 4. 将Pi,ri,u,n导入genetic algorithm 计算w
        from PyOptimize.General_Opt import Test_function

        def Loss(x, P, r, u, n):
            return Test_function().Loss(x, P, r, u, n)

        ga = pyGaft(objfunc=Loss, var_bounds=[(-2, 2)] * 20,
                    individual_size=50, max_iter=200,
                    P=P, r=r, u=u, n=n).run()
        # 5.编写predict3
        # w 从best_fit中获得
        if count == 1:
            import best_fit
        else:
            importlib.reload(best_fit)
        w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
        print('w = ', w)
        rs_pred_y = RankSVM(w=w).predict3(testing_data_X)
        rs_pred_y = np.around(rs_pred_y)
        rs_fpa = PerformanceMeasure(testing_data_y, rs_pred_y).FPA()
        print('rs_fpa:', rs_fpa)

        # RankSVM 效果
        from sklearn.utils import shuffle
        X_shuf, y_shuf = shuffle(training_data_X, training_data_y)
        rs2 = RankSVM().fit(X_shuf, y_shuf)
        rs_pred_y2 = np.around(rs2.predict2(testing_data_X))
        rs_fpa2 = PerformanceMeasure(testing_data_y, rs_pred_y2).FPA()
        rs_aae_result = PerformanceMeasure(testing_data_y, rs_pred_y2).AAE()
        print('rs_fpa2:', rs_fpa2)


if __name__ == '__main__':
    bootstrap()
