from sklearn import linear_model

import numpy as np

from sklearn.linear_model import BayesianRidge

from sklearn.tree import DecisionTreeRegressor

from PerformanceMeasure import PerformanceMeasure

from Processing import Processing

from sklearn.utils import shuffle

from rankSVM import RankSVM

from myGAFT import pyGaft

from RandomUnderSampler import RandomUnderSampler

import importlib

from PyOptimize.General_Opt import Test_function


def Loss(x, **kwargs):
    return Test_function().Loss(x, **kwargs)


def LTR(a, **kwargs):
    return Test_function().LTR(a, **kwargs)


def bootstrap(dataset):

    training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
    ).separate_data(dataset)

    # cost sensitive ranking SVM
    csrs = RankSVM()
    P, r = csrs.transform_pairwise(training_data_X, training_data_y)
    P = P.tolist()
    r = r.tolist()
    u, n = PerformanceMeasure(training_data_y).calc_UN(type='cs')

    count += 1
    global Loss
    csga = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 20,
                  individual_size=100, max_iter=20, max_or_min='min',
                  P=P, r=r, u=u, n=n).run()

    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    print('w = ', w)
    csrs_pred_y = RankSVM(w=w).predict3(testing_data_X)
    csrs_fpa = PerformanceMeasure(testing_data_y, csrs_pred_y).FPA()
    # print('csrs_fpa:', csrs_fpa)
    csrs_fpa_list.append(csrs_fpa)

    # IR SVM
    u, n = PerformanceMeasure(training_data_y).calc_UN(type='ir')

    count += 1
    irga = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 20,
                  individual_size=100, max_iter=20, max_or_min='min',
                  P=P, r=r, u=u, n=n).run()
    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    print('w = ', w)
    irsvm_pred_y = RankSVM(w=w).predict3(testing_data_X)
    irsvm_fpa = PerformanceMeasure(testing_data_y, irsvm_pred_y).FPA()
    # print('irsvm_fpa:', irsvm_fpa)
    irsvm_fpa_list.append(irsvm_fpa)

    # 这个是LTR
    training_datalist_X = training_data_X.tolist()
    training_datalist_y = training_data_y.tolist()

    from PyOptimize.General_Opt import Test_function

    count += 1
    global LTR
    ltrga = pyGaft(objfunc=LTR, var_bounds=[(-2, 2)] * 20,
                   individual_size=50, max_iter=10, max_or_min='max',
                   X=training_datalist_X, y=training_datalist_y).run()

    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    a, fitness_value = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    print('a = {0}'.format(a))
    ltr_pred_y = []
    for test_x in testing_data_X:
        ltr_pred_y.append(np.dot(test_x, a))
    ltr_fpa = PerformanceMeasure(testing_data_y, ltr_pred_y).FPA()
    # print('ltr_fpa', ltr_fpa)
    ltr_fpa_list.append(ltr_fpa)

    # 在原始数据集上训练Ranking SVM,DTR,LR,BRR

    # 这里加上了shuffle，是为了让r的值不全为1，全为1svm会报错
    shuf_X, shuf_y = shuffle(training_data_X, training_data_y)
    rs = RankSVM(C=1.0).fit(shuf_X, shuf_y)
    rs_pred_y = np.around(rs.predict2(testing_data_X))
    rs_fpa = PerformanceMeasure(testing_data_y, rs_pred_y).FPA()
    # print('rs_fpa:', rs_fpa)
    rs_fpa_list.append(rs_fpa)

    dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)
    dtr_pred_y = dtr.predict(testing_data_X)
    dtr_fpa = PerformanceMeasure(testing_data_y, dtr_pred_y).FPA()
    # print('dtr_fpa:', dtr_fpa)
    dtr_fpa_list.append(dtr_fpa)

    lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)
    lr_pred_y = lr.predict(testing_data_X)
    lr_fpa = PerformanceMeasure(testing_data_y, lr_pred_y).FPA()
    # print('lr_fpa:', lr_fpa)
    lr_fpa_list.append(lr_fpa)

    brr = BayesianRidge().fit(training_data_X, training_data_y)
    brr_pred_y = brr.predict(testing_data_X)
    brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
    # print('brr_fpa:', brr_fpa)
    brr_fpa_list.append(brr_fpa)

    # 先对训练数据集进行RUS处理，然后训练Ranking SVM, DTR,LR,BRR
    rus_X, rus_y, _id = RandomUnderSampler(
        ratio=1.0, return_indices=True).fit_sample(training_data_X, training_data_y)

    # LTR
    training_datalist_X = rus_X.tolist()
    training_datalist_y = rus_y.tolist()

    from PyOptimize.General_Opt import Test_function
    count += 1
    rus_ltrga = pyGaft(objfunc=LTR, var_bounds=[(-2, 2)] * 20,
                       individual_size=50, max_iter=10, max_or_min='max',
                       X=training_datalist_X, y=training_datalist_y).run()

    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    rus_a, fitness_value = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    print('rus_a = {0}'.format(rus_a))
    rus_ltr_pred_y = []
    for test_x in testing_data_X:
        rus_ltr_pred_y.append(np.dot(test_x, rus_a))
    rus_ltr_fpa = PerformanceMeasure(testing_data_y, rus_ltr_pred_y).FPA()
    # print('rus_ltr_fpa', rus_ltr_fpa)
    rus_ltr_fpa_list.append(rus_ltr_fpa)

    shuf_X, shuf_y = shuffle(rus_X, rus_y)
    rus_rs = RankSVM(C=1.0).fit(shuf_X, shuf_y)
    rus_rs_pred_y = rus_rs.predict2(testing_data_X)
    rus_rs_fpa = PerformanceMeasure(testing_data_y, rus_rs_pred_y).FPA()
    # print('rus_rs_fpa:', rus_rs_fpa)
    rus_rs_fpa_list.append(rus_rs_fpa)

    rus_dtr = DecisionTreeRegressor().fit(rus_X, rus_y)
    rus_dtr_pred_y = rus_dtr.predict(testing_data_X)
    rus_dtr_fpa = PerformanceMeasure(testing_data_y, rus_dtr_pred_y).FPA()
    # print('rus_dtr_fpa:', rus_dtr_fpa)
    rus_dtr_fpa_list.append(rus_dtr_fpa)

    rus_lr = linear_model.LinearRegression().fit(rus_X, rus_y)
    rus_lr_pred_y = rus_lr.predict(testing_data_X)
    rus_lr_fpa = PerformanceMeasure(testing_data_y, rus_lr_pred_y).FPA()
    # print('rus_lr_fpa:', rus_lr_fpa)
    rus_lr_fpa_list.append(rus_lr_fpa)

    rus_brr = BayesianRidge().fit(rus_X, rus_y)
    rus_brr_pred_y = rus_brr.predict(testing_data_X)
    rus_brr_fpa = PerformanceMeasure(testing_data_y, rus_brr_pred_y).FPA()
    # print('rus_brr_fpa:', rus_brr_fpa)
    rus_brr_fpa_list.append(rus_brr_fpa)


if __name__ == '__main__':

    for dataset, filename in Processing().import_single_data():
        csrs_fpa_list = []
        irsvm_fpa_list = []
        ltr_fpa_list = []
        rs_fpa_list = []
        dtr_fpa_list = []
        lr_fpa_list = []
        brr_fpa_list = []
        rus_ltr_fpa_list = []
        rus_rs_fpa_list = []
        rus_dtr_fpa_list = []
        rus_lr_fpa_list = []
        rus_brr_fpa_list = []

        count = 0
        print ('filename', filename)
        for _ in range(50):
            bootstrap(dataset=dataset)
        csrs_fpa_average = sum(csrs_fpa_list) / 50
        irsvm_fpa_average = sum(irsvm_fpa_list) / 50
        ltr_fpa_average = sum(ltr_fpa_list) / 50
        rs_fpa_average = sum(rs_fpa_list) / 50
        dtr_fpa_average = sum(dtr_fpa_list) / 50
        lr_fpa_average = sum(lr_fpa_list) / 50
        brr_fpa_average = sum(brr_fpa_list) / 50
        rus_ltr_fpa_average = sum(rus_ltr_fpa_list) / 50
        rus_rs_fpa_average = sum(rus_rs_fpa_list) / 50
        rus_dtr_fpa_average = sum(rus_dtr_fpa_list) / 50
        rus_lr_fpa_average = sum(rus_lr_fpa_list) / 50
        rus_brr_fpa_average = sum(rus_brr_fpa_list) / 50

        print('csrs_fpa_list :', csrs_fpa_list)
        print('irsvm_fpa_average :', irsvm_fpa_average)
        print('ltr_fpa_average :', ltr_fpa_average)
        print('rs_fpa_average :', rs_fpa_average)
        print('dtr_fpa_average :', dtr_fpa_average)
        print('lr_fpa_average :', lr_fpa_average)
        print('brr_fpa_average :', brr_fpa_average)
        print('rus_fpa_average :', rus_ltr_fpa_average)
        print('rus_rs_fpa_average :', rus_rs_fpa_average)
        print('rus_dtr_fpa_average :', rus_dtr_fpa_average)
        print('rus_lr_fpa_average :', rus_lr_fpa_average)
        print('rus_brr_fpa_average :', rus_brr_fpa_average)
