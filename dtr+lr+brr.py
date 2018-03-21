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

        dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)
        dtr_pred_y=np.around(dtr.predict(testing_data_X))
        dtr_fpa = PerformanceMeasure(testing_data_y,dtr_pred_y).FPA()
        dtr_aae_result = PerformanceMeasure(testing_data_y, dtr_pred_y).AAE()
        print('dtr_fpa:',dtr_fpa,'dtr_aae_result',dtr_aae_result)


        lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)
        lr_pred_y=np.around(lr.predict(testing_data_X))
        lr_fpa = PerformanceMeasure(testing_data_y,lr_pred_y).FPA()
        lr_aae_result = PerformanceMeasure(testing_data_y, lr_pred_y).AAE()
        print('lr_fpa:',lr_fpa,'lr_aae_result',lr_aae_result)


        brr = BayesianRidge().fit(training_data_X, training_data_y)
        brr_pred_y=np.around(brr.predict(testing_data_X))
        brr_fpa = PerformanceMeasure(testing_data_y,brr_pred_y).FPA()
        brr_aae_result = PerformanceMeasure(testing_data_y, brr_pred_y).AAE()
        print('brr_fpa:',brr_fpa,'brr_aae_result',brr_aae_result)


        smote_X, smote_y = Smote(training_data_X, training_data_y, ratio=1.0, k=5).over_sampling_addorginaldata()

        dtr = DecisionTreeRegressor().fit(smote_X, smote_y)
        dtr_pred_y=np.around(dtr.predict(testing_data_X))
        dtr_fpa = PerformanceMeasure(testing_data_y,dtr_pred_y).FPA()
        dtr_aae_result = PerformanceMeasure(testing_data_y, dtr_pred_y).AAE()
        print('smote_dtr_fpa:',dtr_fpa,'smote_dtr_aae_result',dtr_aae_result)


        lr = linear_model.LinearRegression().fit(smote_X, smote_y)
        lr_pred_y=np.around(lr.predict(testing_data_X))
        lr_fpa = PerformanceMeasure(testing_data_y,lr_pred_y).FPA()
        lr_aae_result = PerformanceMeasure(testing_data_y, lr_pred_y).AAE()
        print('smote_lr_fpa:',lr_fpa,'smote_lr_aae_result',lr_aae_result)


        brr = BayesianRidge().fit(smote_X, smote_y)
        brr_pred_y=np.around(brr.predict(testing_data_X))
        brr_fpa = PerformanceMeasure(testing_data_y,brr_pred_y).FPA()
        brr_aae_result = PerformanceMeasure(testing_data_y, brr_pred_y).AAE()
        print('smote_brr_fpa:',brr_fpa,'smote_brr_aae_result',brr_aae_result)

        rus_X, rus_y, _id = RandomUnderSampler(ratio=1.0, return_indices=True).fit_sample(training_data_X, training_data_y)

        dtr = DecisionTreeRegressor().fit(rus_X, rus_y)
        dtr_pred_y=np.around(dtr.predict(testing_data_X))
        dtr_fpa = PerformanceMeasure(testing_data_y,dtr_pred_y).FPA()
        dtr_aae_result = PerformanceMeasure(testing_data_y, dtr_pred_y).AAE()
        print('rus_dtr_fpa:',dtr_fpa,'rus_dtr_aae_result',dtr_aae_result)


        lr = linear_model.LinearRegression().fit(rus_X, rus_y)
        lr_pred_y=np.around(lr.predict(testing_data_X))
        lr_fpa = PerformanceMeasure(testing_data_y,lr_pred_y).FPA()
        lr_aae_result = PerformanceMeasure(testing_data_y, lr_pred_y).AAE()
        print('rus_lr_fpa:',lr_fpa,'rus_lr_aae_result',lr_aae_result)


        brr = BayesianRidge().fit(rus_X, rus_y)
        brr_pred_y=np.around(brr.predict(testing_data_X))
        brr_fpa = PerformanceMeasure(testing_data_y,brr_pred_y).FPA()
        brr_aae_result = PerformanceMeasure(testing_data_y, brr_pred_y).AAE()
        print('rus_brr_fpa:',brr_fpa,'rus_brr_aae_result',brr_aae_result)


        rng = np.random.RandomState(1)

        dtr = RAdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=rng).fit(
            training_data_X, training_data_y, ratio=1.0)

        dtr_pred_y = np.around(dtr.predict(testing_data_X))
        dtr_fpa = PerformanceMeasure(testing_data_y, dtr_pred_y).FPA()
        dtr_aae_result = PerformanceMeasure(testing_data_y, dtr_pred_y).AAE()
        print('rusboot_dtr_fpa:', dtr_fpa, 'rusboost_dtr_aae_result', dtr_aae_result)

        lr = RAdaBoostRegressor(linear_model.LinearRegression(), n_estimators=100, random_state=rng).fit(
            training_data_X, training_data_y, ratio=1.0)
        lr_pred_y = np.around(lr.predict(testing_data_X))
        lr_fpa = PerformanceMeasure(testing_data_y, lr_pred_y).FPA()
        lr_aae_result = PerformanceMeasure(testing_data_y, lr_pred_y).AAE()
        print('rusboot_lr_fpa:', lr_fpa, 'rusboost_lr_aae_result', lr_aae_result)

        brr = RAdaBoostRegressor(BayesianRidge(), n_estimators=100, random_state=rng).fit(training_data_X,
                                                                                                 training_data_y,
                                                                                                 ratio=1.0)
        brr_pred_y = np.around(brr.predict(testing_data_X))
        brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
        brr_aae_result = PerformanceMeasure(testing_data_y, brr_pred_y).AAE()
        print('rusboot_brr_fpa:', brr_fpa, 'rusboost_brr_aae_result', brr_aae_result)



        dtr = SAdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=100, random_state=rng).fit(
            training_data_X, training_data_y,ratio=1.0)
        dtr_pred_y = np.around(dtr.predict(testing_data_X))
        dtr_fpa = PerformanceMeasure(testing_data_y, dtr_pred_y).FPA()
        dtr_aae_result = PerformanceMeasure(testing_data_y, dtr_pred_y).AAE()
        print('smoteboot_dtr_fpa:', dtr_fpa, 'smoteboost_dtr_aae_result', dtr_aae_result)


        lr = SAdaBoostRegressor(linear_model.LinearRegression(), n_estimators=100, random_state=rng).fit(
             training_data_X, training_data_y, ratio=1.0)
        lr_pred_y = np.around(lr.predict(testing_data_X))
        lr_fpa = PerformanceMeasure(testing_data_y, lr_pred_y).FPA()
        lr_aae_result = PerformanceMeasure(testing_data_y, lr_pred_y).AAE()
        print('smoteboot_lr_fpa:', lr_fpa, 'smoteboost_lr_aae_result', lr_aae_result)

        brr = SAdaBoostRegressor(BayesianRidge(), n_estimators=100, random_state=rng).fit(
            training_data_X,training_data_y,ratio=1.0)
        brr_pred_y=np.around(brr.predict(testing_data_X))
        brr_fpa = PerformanceMeasure(testing_data_y,brr_pred_y).FPA()
        brr_aae_result = PerformanceMeasure(testing_data_y, brr_pred_y).AAE()
        print('smoteboot_brr_fpa:',brr_fpa,'smoteboot_brr_aae_result',brr_aae_result)
    


if __name__ == '__main__':
    bootstrap()










