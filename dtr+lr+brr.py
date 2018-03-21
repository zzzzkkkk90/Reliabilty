from sklearn import linear_model

import numpy as np

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing

from Smote import Smote

from RandomUnderSampler import RandomUnderSampler

from RusAdaBoostRegressor import RAdaBoostRegressor

from SmoteAdaBoostRegressor import SAdaBoostRegressor





def bootstrap():

    #dataset = Processing().import_data()
   for dataset, filename in Processing().import_single_data():

        training_data_X, training_data_y, testing_data_X, testing_data_y= Processing().separate_data(dataset)



        lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)
        lr_pred_y=np.around(lr.predict(testing_data_X))
        lr_fpa = PerformanceMeasure(testing_data_y,lr_pred_y).FPA()
        lr_aae_result = PerformanceMeasure(testing_data_y, lr_pred_y).AAE()
        print('lr_fpa:',lr_fpa,'lr_aae_result',lr_aae_result)
    

if __name__ == '__main__':
    bootstrap()










