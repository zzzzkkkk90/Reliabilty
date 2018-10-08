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

from Smote import Smote

import importlib

from PyOptimize.General_Opt import Test_function


def Loss(x, **kwargs):
    return Test_function().Loss(x, **kwargs)


def LTR(a, **kwargs):
    return Test_function().LTR(a, **kwargs)


def bootstrap(training_data_X, training_data_y, testing_data_X, testing_data_y, dataset, trainingfilename, testingfilename):

    trainingdataname.append(trainingfilename)
    print('trainingdata',trainingfilename)
    count = 0
    # cost sensitive ranking SVM
    csrs = RankSVM()
    P, r = csrs.transform_pairwise(training_data_X, training_data_y)
    P = P.tolist()
    r = r.tolist()
    u, n = PerformanceMeasure(training_data_y).calc_UN(type='cs')
    count += 1
    global Loss
    csga = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 20,
                  individual_size=500, max_iter=200, max_or_min='min',
                  P=P, r=r, u=u, n=n).run()
    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    csrs_pred_y = RankSVM(w=w).predict3(testing_data_X)
    csrs_fpa = PerformanceMeasure(testing_data_y, csrs_pred_y).FPA()
    csrs_pofb = PerformanceMeasure(testing_data_y, csrs_pred_y).PofB20()
    csrs_fpa_list.append(csrs_fpa)
    csrs_pofb_list.append(csrs_pofb)


    # IR SVM
    u, n = PerformanceMeasure(training_data_y).calc_UN(type='ir')

    count += 1
    irga = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 20,
                  individual_size=500, max_iter=200, max_or_min='min',
                  P=P, r=r, u=u, n=n).run()
    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    irsvm_pred_y = RankSVM(w=w).predict3(testing_data_X)
    irsvm_fpa = PerformanceMeasure(testing_data_y, irsvm_pred_y).FPA()
    irsvm_pofb = PerformanceMeasure(testing_data_y, irsvm_pred_y).PofB20()
    irsvm_fpa_list.append(irsvm_fpa)
    irsvm_pofb_list.append(irsvm_pofb)

    # 这里还要加个去掉另一个参数的
    u, n = PerformanceMeasure(training_data_y).calc_UN(type='svm')

    count += 1
    irga = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 20,
                  individual_size=500, max_iter=200, max_or_min='min',
                  P=P, r=r, u=u, n=n).run()
    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    #print('w = ', w)
    svm_pred_y = RankSVM(w=w).predict3(testing_data_X)
    svm_fpa = PerformanceMeasure(testing_data_y, svm_pred_y).FPA()
    svm_pofb = PerformanceMeasure(testing_data_y, svm_pred_y).PofB20()
    svm_fpa_list.append(svm_fpa)
    svm_pofb_list.append(svm_pofb)

    # 这个是LTR
    training_datalist_X = training_data_X.tolist()
    training_datalist_y = training_data_y.tolist()

    from PyOptimize.General_Opt import Test_function

    count += 1
    global LTR
    ltrga = pyGaft(objfunc=LTR, var_bounds=[(-20, 20)] * 20,
                   individual_size=100, max_iter=200, max_or_min='max',
                   X=training_datalist_X, y=training_datalist_y).run()

    if count == 1:
        import best_fit
    else:
        importlib.reload(best_fit)

    a, fitness_value = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
    #print('a = {0}'.format(a))
    ltr_pred_y = []
    for test_x in testing_data_X:
        ltr_pred_y.append(np.dot(test_x, a))
    ltr_fpa = PerformanceMeasure(testing_data_y, ltr_pred_y).FPA()
    ltr_pofb = PerformanceMeasure(testing_data_y, ltr_pred_y).PofB20()
    ltr_fpa_list.append(ltr_fpa)
    ltr_pofb_list.append(ltr_pofb)

    # 在原始数据集上训练Ranking SVM,DTR,LR,BRR

    # 这里加上了shuffle，是为了让r的值不全为1，全为1svm会报错
    shuf_X, shuf_y = shuffle(training_data_X, training_data_y)
    rs = RankSVM(C=1.0).fit(shuf_X, shuf_y)
    rs_pred_y = np.around(rs.predict2(testing_data_X))
    rs_fpa = PerformanceMeasure(testing_data_y, rs_pred_y).FPA()
    rs_pofb = PerformanceMeasure(testing_data_y, rs_pred_y).PofB20()
    rs_fpa_list.append(rs_fpa)
    rs_pofb_list.append(rs_pofb)

    dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)
    dtr_pred_y = dtr.predict(testing_data_X)
    dtr_fpa = PerformanceMeasure(testing_data_y, dtr_pred_y).FPA()
    dtr_pofb = PerformanceMeasure(testing_data_y, dtr_pred_y).PofB20()
    dtr_fpa_list.append(dtr_fpa)
    dtr_pofb_list.append(dtr_pofb)

    lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)
    lr_pred_y = lr.predict(testing_data_X)
    lr_fpa = PerformanceMeasure(testing_data_y, lr_pred_y).FPA()
    lr_pofb = PerformanceMeasure(testing_data_y, lr_pred_y).PofB20()
    lr_fpa_list.append(lr_fpa)
    lr_pofb_list.append(lr_pofb)

    brr = BayesianRidge().fit(training_data_X, training_data_y)
    brr_pred_y = brr.predict(testing_data_X)
    brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
    brr_pofb = PerformanceMeasure(testing_data_y, brr_pred_y).PofB20()
    brr_fpa_list.append(brr_fpa)
    brr_pofb_list.append(brr_pofb)

if __name__ == '__main__':

    csrs_fpa_list = []
    irsvm_fpa_list = []
    svm_fpa_list = []
    ltr_fpa_list = []
    rs_fpa_list = []
    dtr_fpa_list = []
    lr_fpa_list = []
    brr_fpa_list = []

    csrs_pofb_list = []
    irsvm_pofb_list = []
    svm_pofb_list = []
    ltr_pofb_list = []
    rs_pofb_list = []
    dtr_pofb_list = []
    lr_pofb_list = []
    brr_pofb_list = []

    trainingdataname = []

    for training_data_X, training_data_y, testing_data_X, testing_data_y, dataset, trainingfilename, testingfilename in Processing().import_single_data_withduo():
        bootstrap(training_data_X, training_data_y, testing_data_X, testing_data_y,dataset, trainingfilename, testingfilename)

    file = open('cvresult.txt', 'a')

    file.write(str(trainingdataname))
    file.write("\n")

    file.write('csrs_fpa_list :' + str(csrs_fpa_list))
    file.write("\n")

    file.write('irsvm_fpa_list :' + str(irsvm_fpa_list))
    file.write("\n")

    file.write('svm_fpa_list :' + str(svm_fpa_list))
    file.write("\n")

    file.write('ltr_fpa_list :' + str(ltr_fpa_list))
    file.write("\n")

    file.write('rs_fpa_list :' + str(rs_fpa_list))
    file.write("\n")

    file.write('dtr_fpa_list :' + str(dtr_fpa_list))
    file.write("\n")

    file.write('lr_fpa_list :' + str(lr_fpa_list))
    file.write("\n")

    file.write('brr_fpa_list :' + str(brr_fpa_list))
    file.write("\n")

    file.write('csrs_pofb_list :' + str(csrs_pofb_list))
    file.write("\n")

    file.write('irsvm_pofb_list :' + str(irsvm_pofb_list))
    file.write("\n")

    file.write('svm_pofb_list :' + str(svm_pofb_list))
    file.write("\n")

    file.write('ltr_pofb_list :' + str(ltr_pofb_list))
    file.write("\n")

    file.write('rs_pofb_list :' + str(rs_pofb_list))
    file.write("\n")

    file.write('dtr_pofb_list :' + str(dtr_fpa_list))
    file.write("\n")

    file.write('lr_pofb_list :' + str(lr_pofb_list))
    file.write("\n")

    file.write('brr_pofb_list :' + str(brr_pofb_list))
    file.write("\n")

    file.close()


