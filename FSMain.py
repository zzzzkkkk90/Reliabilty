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

from IG import IG


def Loss(x, **kwargs):
    return Test_function().Loss(x, **kwargs)


def LTR(a, **kwargs):
    return Test_function().LTR(a, **kwargs)


def bootstrap(dataset):

    training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
    ).separate_data(dataset)

    training_data_X2, training_data_y2, testing_data_X2, testing_data_y2 = IG(training_data_X, training_data_y,
                                                                              testing_data_X,
                                                                              testing_data_y).getSelectedFeature(2)

    training_data_X3, training_data_y3, testing_data_X3, testing_data_y3 = IG(training_data_X, training_data_y,
                                                                              testing_data_X,
                                                                              testing_data_y).getSelectedFeature(3)

    training_data_X5, training_data_y5, testing_data_X5, testing_data_y5 = IG(training_data_X, training_data_y,
                                                                              testing_data_X,
                                                                              testing_data_y).getSelectedFeature(5)

    training_data_X8, training_data_y8, testing_data_X8, testing_data_y8 = IG(training_data_X, training_data_y,
                                                                              testing_data_X,
                                                                              testing_data_y).getSelectedFeature(8)
    training_data_X13, training_data_y13, testing_data_X13, testing_data_y13 = IG(training_data_X, training_data_y,
                                                                                  testing_data_X,
                                                                                  testing_data_y).getSelectedFeature(13)

    # cost sensitive ranking SVM with the number of features as 2
    csrs2 = RankSVM()
    P, r = csrs2.transform_pairwise(training_data_X2, training_data_y2)
    P = P.tolist()
    r = r.tolist()
    u, n = PerformanceMeasure(training_data_y2).calc_UN(type='cs')

    global Loss
    csga2 = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 2,
                   individual_size=500, max_iter=200, max_or_min='min',
                   P=P, r=r, u=u, n=n).run()

    importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    csrs_pred_y2 = RankSVM(w=w).predict3(testing_data_X2)
    csrs_fpa2 = PerformanceMeasure(testing_data_y2, csrs_pred_y2).FPA()
    # print('csrs_fpa:', csrs_fpa)
    csrs_fpa_list2.append(csrs_fpa2)

    # cost sensitive ranking SVM with the number of features as 3
    csrs3 = RankSVM()
    P, r = csrs3.transform_pairwise(training_data_X3, training_data_y3)
    P = P.tolist()
    r = r.tolist()
    u, n = PerformanceMeasure(training_data_y3).calc_UN(type='cs')

    first = False
    csga3 = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 3,
                   individual_size=500, max_iter=200, max_or_min='min',
                   P=P, r=r, u=u, n=n).run()

    importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    # print('w = ', w)
    csrs_pred_y3 = RankSVM(w=w).predict3(testing_data_X3)
    csrs_fpa3 = PerformanceMeasure(testing_data_y3, csrs_pred_y3).FPA()
    # print('irsvm_fpa:', irsvm_fpa)
    csrs_fpa_list3.append(csrs_fpa3)

    # cost sensitive ranking SVM with the number of features as 5
    csrs5 = RankSVM()
    P, r = csrs5.transform_pairwise(training_data_X5, training_data_y5)
    P = P.tolist()
    r = r.tolist()
    u, n = PerformanceMeasure(training_data_y5).calc_UN(type='cs')

    csga5 = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 5,
                   individual_size=500, max_iter=200, max_or_min='min',
                   P=P, r=r, u=u, n=n).run()
    # if count == 1:
    #     import best_fit
    # else:
    importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    # print('w = ', w)
    csrs_pred_y5 = RankSVM(w=w).predict3(testing_data_X5)
    csrs_fpa5 = PerformanceMeasure(testing_data_y5, csrs_pred_y5).FPA()
    # print('irsvm_fpa:', irsvm_fpa)
    csrs_fpa_list5.append(csrs_fpa5)

    # cost sensitive ranking SVM with the number of features as 8
    csrs8 = RankSVM()
    P, r = csrs8.transform_pairwise(training_data_X8, training_data_y8)
    P = P.tolist()
    r = r.tolist()
    u, n = PerformanceMeasure(training_data_y8).calc_UN(type='cs')

    csga8 = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 8,
                   individual_size=500, max_iter=200, max_or_min='min',
                   P=P, r=r, u=u, n=n).run()

    importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    # print('w = ', w)
    csrs_pred_y8 = RankSVM(w=w).predict3(testing_data_X8)
    csrs_fpa8 = PerformanceMeasure(testing_data_y8, csrs_pred_y8).FPA()
    # print('irsvm_fpa:', irsvm_fpa)
    csrs_fpa_list8.append(csrs_fpa8)

    # cost sensitive ranking SVM with the number of features as 13
    csrs13 = RankSVM()
    P, r = csrs13.transform_pairwise(training_data_X13, training_data_y13)
    P = P.tolist()
    r = r.tolist()
    u, n = PerformanceMeasure(training_data_y13).calc_UN(type='cs')

    csga13 = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 13,
                    individual_size=500, max_iter=200, max_or_min='min',
                    P=P, r=r, u=u, n=n).run()

    importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    # print('w = ', w)
    csrs_pred_y13 = RankSVM(w=w).predict3(testing_data_X13)
    csrs_fpa13 = PerformanceMeasure(testing_data_y13, csrs_pred_y13).FPA()
    # print('irsvm_fpa:', irsvm_fpa)
    csrs_fpa_list13.append(csrs_fpa13)
    print(f'first={first}')


if __name__ == '__main__':
    with open('best_fit.py', 'w') as f:
        import best_fit

    for dataset, filename in Processing().import_single_data():
        csrs_fpa_list2 = []
        csrs_fpa_list3 = []
        csrs_fpa_list5 = []
        csrs_fpa_list8 = []
        csrs_fpa_list13 = []

        for _ in range(5):
            bootstrap(dataset=dataset)

        print('filename', filename)
        csrs_fpa_average2 = sum(csrs_fpa_list2) / 5
        csrs_fpa_average3 = sum(csrs_fpa_list3) / 5
        csrs_fpa_average5 = sum(csrs_fpa_list5) / 5
        csrs_fpa_average8 = sum(csrs_fpa_list8) / 5
        csrs_fpa_average13 = sum(csrs_fpa_list13) / 5

        print('csrs_fpa_list2 :', csrs_fpa_list2)
        print('csrs_fpa_list3 :', csrs_fpa_list3)
        print('csrs_fpa_list5 :', csrs_fpa_list5)
        print('csrs_fpa_list8 :', csrs_fpa_list8)
        print('csrs_fpa_list13 :', csrs_fpa_list13)

        file = open('result.txt', 'a')
        file.write(filename)
        file.write("\n")

        file.write('csrs_fpa_list2 :' + str(csrs_fpa_average2))
        file.write("\n")

        file.write('csrs_fpa_list3 :' + str(csrs_fpa_average3))
        file.write("\n")

        file.write('csrs_fpa_list5 :' + str(csrs_fpa_average5))
        file.write("\n")

        file.write('csrs_fpa_list8 :' + str(csrs_fpa_average8))
        file.write("\n")

        file.write('csrs_fpa_list13 :' + str(csrs_fpa_average13))
        file.write("\n")

        file.close()
