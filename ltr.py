# -*- coding: utf-8 -*-
# @Author: Kicc Shen
# @Date:   2018-04-12 19:36:53
# @Last Modified by:   kicc
# @Last Modified time: 2018-04-22 19:54:21
import numpy as np

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing


from myGAFT import pyGaft

import importlib


def bootstrap():

    #dataset = Processing().import_data()
    count = 0
    for dataset, filename in Processing().import_single_data():
        print(filename)
        count += 1
        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)

        # print('train shape', training_data_X.shape)
        training_data_X = training_data_X.tolist()
        training_data_y = training_data_y.tolist()

        from PyOptimize.General_Opt import Test_function

        def LTR(a, X, y):
            return Test_function().LTR(a, X, y)

        ga = pyGaft(objfunc=LTR, var_bounds=[(-2, 2)] * 20,
                    individual_size=50, max_iter=10,
                    X=training_data_X, y=training_data_y).run()

        if count == 1:
            import best_fit
        else:
            importlib.reload(best_fit)

        a, fitness_value = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
        print('a = {0}'.format(a))
        pred_y = []
        for test_x in testing_data_X:
            pred_y.append(np.dot(test_x, a))

        fpa = PerformanceMeasure(testing_data_y, pred_y).FPA()

        print('fpa = {0}'.format(fpa))


if __name__ == '__main__':
    bootstrap()
