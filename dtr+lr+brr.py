from sklearn import linear_model

import numpy as np

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing

from SmoteYU import Smote

from RandomUnderSampler import RandomUnderSampler

from RusAdaBoostRegressor import RAdaBoostRegressor

from SmoteAdaBoostRegressor import SAdaBoostRegressor

from rankSVM import RankSVM


def bootstrap():

    #dataset = Processing().import_data()
    for dataset, filename in Processing().import_single_data():

        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)

        rs = RankSVM().fit(training_data_X, training_data_y)
        rs_pred_y = np.around(rs.predict2(testing_data_X))
        rs_fpa = PerformanceMeasure(testing_data_y, rs_pred_y).FPA()
        rs_aae_result = PerformanceMeasure(testing_data_y, rs_pred_y).AAE()
        print('rs_fpa:', rs_fpa, 'rs_aae_result', rs_aae_result)

        lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)
        lr_pred_y = np.around(lr.predict(testing_data_X))
        lr_fpa = PerformanceMeasure(testing_data_y, lr_pred_y).FPA()
        lr_aae_result = PerformanceMeasure(testing_data_y, lr_pred_y).AAE()
        print('lr_fpa:', lr_fpa, 'lr_aae_result', lr_aae_result)


if __name__ == '__main__':
    bootstrap()
