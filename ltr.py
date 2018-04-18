import numpy as np

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing


from myGAFT import pyGaft

import importlib


def bootstrap():

    #dataset = Processing().import_data()
    for dataset, filename in Processing().import_single_data():
        print(filename)
        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)

        # print('train shape', training_data_X.shape)
        training_data_X = training_data_X.tolist()
        training_data_y = training_data_y.tolist()

        from PyOptimize.General_Opt import Test_function

        def LTR(a, X, y):
            return Test_function().LTR(a, X, y)

        ga = pyGaft(objfunc=LTR, var_bounds=[(-2, 2)] * 20,
                    individual_size=50, max_iter=500,
                    X=training_data_X, y=training_data_y).run()


if __name__ == '__main__':
    bootstrap()
