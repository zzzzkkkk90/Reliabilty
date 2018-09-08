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


def bootstrap(dataset):

        count = 0
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
                      individual_size=500, max_iter=2, max_or_min='min',
                      P=P, r=r, u=u, n=n).run()

        if count == 1:
            import best_fit
        else:
            importlib.reload(best_fit)

        w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
        #print('w = ', w)
        csrs_pred_y = RankSVM(w=w).predict3(testing_data_X)
        csrs_fpa = PerformanceMeasure(testing_data_y, csrs_pred_y).FPA()
        csrs_fpa_list.append(csrs_fpa)

        csrs_pofb=PerformanceMeasure(testing_data_y, csrs_pred_y).PofB20()
        csrs_pofd = PerformanceMeasure(testing_data_y, csrs_pred_y).PofD20()
        csrs_ranking = PerformanceMeasure(testing_data_y, csrs_pred_y).ranking()
        csrs_pofb_list.append(csrs_pofb)
        csrs_pofd_list.append(csrs_pofd)


        # IR SVM
        u, n = PerformanceMeasure(training_data_y).calc_UN(type='ir')

        count += 1
        irga = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 20,
                      individual_size=500, max_iter=2, max_or_min='min',
                      P=P, r=r, u=u, n=n).run()
        if count == 1:
            import best_fit
        else:
            importlib.reload(best_fit)

        w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
        #print('w = ', w)
        irsvm_pred_y = RankSVM(w=w).predict3(testing_data_X)
        irsvm_fpa = PerformanceMeasure(testing_data_y, irsvm_pred_y).FPA()
        #print('irsvm_fpa:', irsvm_fpa)
        irsvm_fpa_list.append(irsvm_fpa)

        irsvm_pofb = PerformanceMeasure(testing_data_y, irsvm_pred_y).PofB20()
        irsvm_pofd = PerformanceMeasure(testing_data_y, irsvm_pred_y).PofD20()
        irsvm_ranking = PerformanceMeasure(testing_data_y,irsvm_pred_y).ranking()
        irsvm_pofb_list.append(irsvm_pofb)
        irsvm_pofd_list.append(irsvm_pofd)

        #这里还要加个去掉另一个参数的
        u, n = PerformanceMeasure(training_data_y).calc_UN(type='svm')

        count += 1
        irga = pyGaft(objfunc=Loss, var_bounds=[(-1, 1)] * 20,
                      individual_size=500, max_iter=2, max_or_min='min',
                      P=P, r=r, u=u, n=n).run()
        if count == 1:
            import best_fit
        else:
            importlib.reload(best_fit)

        w, fitness = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
        #print('w = ', w)
        svm_pred_y = RankSVM(w=w).predict3(testing_data_X)
        svm_fpa = PerformanceMeasure(testing_data_y, svm_pred_y).FPA()
        #print('svm_fpa:', svm_fpa)
        svm_fpa_list.append(svm_fpa)

        svm_pofb = PerformanceMeasure(testing_data_y, svm_pred_y).PofB20()
        svm_pofd = PerformanceMeasure(testing_data_y, svm_pred_y).PofD20()
        svm_ranking = PerformanceMeasure(testing_data_y,svm_pred_y).ranking()
        svm_pofb_list.append(svm_pofb)
        svm_pofd_list.append(svm_pofd)


        # 这个是LTR
        training_datalist_X = training_data_X.tolist()
        training_datalist_y = training_data_y.tolist()

        from PyOptimize.General_Opt import Test_function

        count += 1
        global LTR
        ltrga = pyGaft(objfunc=LTR, var_bounds=[(-20, 20)] * 20,
                       individual_size=100, max_iter=2, max_or_min='max',
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
        # print('ltr_fpa', ltr_fpa)
        ltr_fpa_list.append(ltr_fpa)

        ltr_pofb = PerformanceMeasure(testing_data_y, ltr_pred_y).PofB20()
        ltr_pofd = PerformanceMeasure(testing_data_y,ltr_pred_y).PofD20()
        ltr_ranking = PerformanceMeasure(testing_data_y, ltr_pred_y).ranking()
        ltr_pofb_list.append(ltr_pofb)
        ltr_pofd_list.append(ltr_pofd)

        # 在原始数据集上训练Ranking SVM,DTR,LR,BRR

        # 这里加上了shuffle，是为了让r的值不全为1，全为1svm会报错
        shuf_X, shuf_y = shuffle(training_data_X, training_data_y)
        rs = RankSVM(C=1.0).fit(shuf_X, shuf_y)
        rs_pred_y = np.around(rs.predict2(testing_data_X))
        rs_fpa = PerformanceMeasure(testing_data_y, rs_pred_y).FPA()
        # print('rs_fpa:', rs_fpa)
        rs_fpa_list.append(rs_fpa)

        rs_pofb = PerformanceMeasure(testing_data_y, rs_pred_y).PofB20()
        rs_pofd = PerformanceMeasure(testing_data_y,rs_pred_y).PofD20()
        rs_ranking = PerformanceMeasure(testing_data_y, rs_pred_y).ranking()
        rs_pofb_list.append(rs_pofb)
        rs_pofd_list.append(rs_pofd)

        dtr = DecisionTreeRegressor().fit(training_data_X, training_data_y)
        dtr_pred_y = dtr.predict(testing_data_X)
        dtr_fpa = PerformanceMeasure(testing_data_y, dtr_pred_y).FPA()
        # print('dtr_fpa:', dtr_fpa)
        dtr_fpa_list.append(dtr_fpa)

        dtr_pofb = PerformanceMeasure(testing_data_y, dtr_pred_y).PofB20()
        dtr_pofd = PerformanceMeasure(testing_data_y,dtr_pred_y).PofD20()
        dtr_ranking = PerformanceMeasure(testing_data_y, dtr_pred_y).ranking()
        dtr_pofb_list.append(dtr_pofb)
        dtr_pofd_list.append(dtr_pofd)

        lr = linear_model.LinearRegression().fit(training_data_X, training_data_y)
        lr_pred_y = lr.predict(testing_data_X)
        lr_fpa = PerformanceMeasure(testing_data_y, lr_pred_y).FPA()
        # print('lr_fpa:', lr_fpa)
        lr_fpa_list.append(lr_fpa)

        lr_pofb = PerformanceMeasure(testing_data_y, lr_pred_y).PofB20()
        lr_pofd = PerformanceMeasure(testing_data_y,lr_pred_y).PofD20()
        lr_ranking = PerformanceMeasure(testing_data_y, lr_pred_y).ranking()
        lr_pofb_list.append(lr_pofb)
        lr_pofd_list.append(lr_pofd)

        brr = BayesianRidge().fit(training_data_X, training_data_y)
        brr_pred_y = brr.predict(testing_data_X)
        brr_fpa = PerformanceMeasure(testing_data_y, brr_pred_y).FPA()
        # print('brr_fpa:', brr_fpa)
        brr_fpa_list.append(brr_fpa)

        brr_pofb = PerformanceMeasure(testing_data_y, brr_pred_y).PofB20()
        brr_pofd = PerformanceMeasure(testing_data_y,brr_pred_y).PofD20()
        brr_ranking = PerformanceMeasure(testing_data_y, brr_pred_y).ranking()
        brr_pofb_list.append(brr_pofb)
        brr_pofd_list.append(brr_pofd)




        # 先对训练数据集进行RUS处理，然后训练Ranking SVM, DTR,LR,BRR
        rus_X, rus_y, _id = RandomUnderSampler(
            ratio=1.0, return_indices=True).fit_sample(training_data_X, training_data_y)

        # LTR
        training_datalist_X = rus_X.tolist()
        training_datalist_y = rus_y.tolist()

        from PyOptimize.General_Opt import Test_function
        count += 1
        rus_ltrga = pyGaft(objfunc=LTR, var_bounds=[(-20, 20)] * 20,
                           individual_size=100, max_iter=2, max_or_min='max',
                           X=training_datalist_X, y=training_datalist_y).run()

        if count == 1:
            import best_fit
        else:
            importlib.reload(best_fit)

        rus_a, fitness_value = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]
        #print('rus_a = {0}'.format(rus_a))
        rus_ltr_pred_y = []
        for test_x in testing_data_X:
            rus_ltr_pred_y.append(np.dot(test_x, rus_a))
        rus_ltr_fpa = PerformanceMeasure(testing_data_y, rus_ltr_pred_y).FPA()
        # print('rus_ltr_fpa', rus_ltr_fpa)
        rus_ltr_fpa_list.append(rus_ltr_fpa)

        rus_ltr_pofb = PerformanceMeasure(testing_data_y, rus_ltr_pred_y).PofB20()
        rus_ltr_pofd = PerformanceMeasure(testing_data_y,rus_ltr_pred_y).PofD20()
        rus_ltr_ranking = PerformanceMeasure(testing_data_y, rus_ltr_pred_y).ranking()
        rus_ltr_pofb_list.append(rus_ltr_pofb)
        rus_ltr_pofd_list.append(rus_ltr_pofd)


        shuf_X, shuf_y = shuffle(rus_X, rus_y)
        rus_rs = RankSVM(C=1.0).fit(shuf_X, shuf_y)
        rus_rs_pred_y = rus_rs.predict2(testing_data_X)
        rus_rs_fpa = PerformanceMeasure(testing_data_y, rus_rs_pred_y).FPA()
        # print('rus_rs_fpa:', rus_rs_fpa)
        rus_rs_fpa_list.append(rus_rs_fpa)

        rus_rs_pofb = PerformanceMeasure(testing_data_y, rus_rs_pred_y).PofB20()
        rus_rs_pofd = PerformanceMeasure(testing_data_y,rus_rs_pred_y).PofD20()
        rus_rs_ranking = PerformanceMeasure(testing_data_y, rus_rs_pred_y).ranking()
        rus_rs_pofb_list.append(rus_rs_pofb)
        rus_rs_pofd_list.append(rus_rs_pofd)



        rus_dtr = DecisionTreeRegressor().fit(rus_X, rus_y)
        rus_dtr_pred_y = rus_dtr.predict(testing_data_X)
        rus_dtr_fpa = PerformanceMeasure(testing_data_y, rus_dtr_pred_y).FPA()
        # print('rus_dtr_fpa:', rus_dtr_fpa)
        rus_dtr_fpa_list.append(rus_dtr_fpa)

        rus_dtr_pofb = PerformanceMeasure(testing_data_y, rus_dtr_pred_y).PofB20()
        rus_dtr_pofd = PerformanceMeasure(testing_data_y,rus_dtr_pred_y).PofD20()
        rus_dtr_ranking = PerformanceMeasure(testing_data_y, rus_dtr_pred_y).ranking()
        rus_dtr_pofb_list.append(rus_dtr_pofb)
        rus_dtr_pofd_list.append(rus_dtr_pofd)


        rus_lr = linear_model.LinearRegression().fit(rus_X, rus_y)
        rus_lr_pred_y = rus_lr.predict(testing_data_X)
        rus_lr_fpa = PerformanceMeasure(testing_data_y, rus_lr_pred_y).FPA()
        # print('rus_lr_fpa:', rus_lr_fpa)
        rus_lr_fpa_list.append(rus_lr_fpa)

        rus_lr_pofb = PerformanceMeasure(testing_data_y, rus_lr_pred_y).PofB20()
        rus_lr_pofd = PerformanceMeasure(testing_data_y,rus_lr_pred_y).PofD20()
        rus_lr_ranking = PerformanceMeasure(testing_data_y, rus_lr_pred_y).ranking()
        rus_lr_pofb_list.append(rus_lr_pofb)
        rus_lr_pofd_list.append(rus_lr_pofd)

        rus_brr = BayesianRidge().fit(rus_X, rus_y)
        rus_brr_pred_y = rus_brr.predict(testing_data_X)
        rus_brr_fpa = PerformanceMeasure(testing_data_y, rus_brr_pred_y).FPA()
        # print('rus_brr_fpa:', rus_brr_fpa)
        rus_brr_fpa_list.append(rus_brr_fpa)

        rus_brr_pofb = PerformanceMeasure(testing_data_y, rus_brr_pred_y).PofB20()
        rus_brr_pofd = PerformanceMeasure(testing_data_y,rus_brr_pred_y).PofD20()
        rus_brr_ranking = PerformanceMeasure(testing_data_y, rus_brr_pred_y).ranking()
        rus_brr_pofb_list.append(rus_brr_pofb)
        rus_brr_pofd_list.append(rus_brr_pofd)




        # 先对训练数据集进行Smote处理，然后训练Ranking SVM, DTR,LR,BRR
        smote_X, smote_y= Smote(training_data_X, training_data_y, ratio=1.0, k=5).over_sampling_addorginaldata()

        # LTR
        training_datalist_X = smote_X.tolist()
        training_datalist_y = smote_y.tolist()

        from PyOptimize.General_Opt import Test_function
        count += 1
        smote_ltrga = pyGaft(objfunc=LTR, var_bounds=[(-20, 20)] * 20,
                           individual_size=100, max_iter=2, max_or_min='max',
                           X=training_datalist_X, y=training_datalist_y).run()

        if count == 1:
            import best_fit
        else:
            importlib.reload(best_fit)

        smote_a, fitness_value = best_fit.best_fit[-1][-2], best_fit.best_fit[-1][-1]

        smote_ltr_pred_y = []
        for test_x in testing_data_X:
            smote_ltr_pred_y.append(np.dot(test_x, smote_a))
        smote_ltr_fpa = PerformanceMeasure(testing_data_y, smote_ltr_pred_y).FPA()

        smote_ltr_fpa_list.append(smote_ltr_fpa)

        smote_ltr_pofb = PerformanceMeasure(testing_data_y, smote_ltr_pred_y).PofB20()
        smote_ltr_pofd = PerformanceMeasure(testing_data_y,smote_ltr_pred_y).PofD20()
        smote_ltr_ranking = PerformanceMeasure(testing_data_y, smote_ltr_pred_y).ranking()
        smote_ltr_pofb_list.append(smote_ltr_pofb)
        smote_ltr_pofd_list.append(smote_ltr_pofd)


        shuf_X, shuf_y = shuffle(smote_X, smote_y)

        smote_rs = RankSVM(C=1.0).fit(shuf_X, shuf_y)
        smote_rs_pred_y = smote_rs.predict2(testing_data_X)
        smote_rs_fpa = PerformanceMeasure(testing_data_y, smote_rs_pred_y).FPA()
        smote_rs_fpa_list.append(smote_rs_fpa)
        smote_rs_pofb = PerformanceMeasure(testing_data_y, smote_rs_pred_y).PofB20()
        smote_rs_pofd = PerformanceMeasure(testing_data_y,smote_rs_pred_y).PofD20()
        smote_rs_ranking = PerformanceMeasure(testing_data_y, smote_rs_pred_y).ranking()
        smote_rs_pofb_list.append(smote_rs_pofb)
        smote_rs_pofd_list.append(smote_rs_pofd)

        smote_dtr = DecisionTreeRegressor().fit(smote_X, smote_y)
        smote_dtr_pred_y = smote_dtr.predict(testing_data_X)
        smote_dtr_fpa = PerformanceMeasure(testing_data_y, smote_dtr_pred_y).FPA()
        smote_dtr_fpa_list.append(smote_dtr_fpa)
        smote_dtr_pofb = PerformanceMeasure(testing_data_y, smote_dtr_pred_y).PofB20()
        smote_dtr_pofd = PerformanceMeasure(testing_data_y,smote_dtr_pred_y).PofD20()
        smote_dtr_ranking = PerformanceMeasure(testing_data_y, smote_dtr_pred_y).ranking()
        smote_dtr_pofb_list.append(smote_dtr_pofb)
        smote_dtr_pofd_list.append(smote_dtr_pofd)

        smote_lr = linear_model.LinearRegression().fit(smote_X, smote_y)
        smote_lr_pred_y = smote_lr.predict(testing_data_X)
        smote_lr_fpa = PerformanceMeasure(testing_data_y, smote_lr_pred_y).FPA()
        smote_lr_fpa_list.append(smote_lr_fpa)
        smote_lr_pofb = PerformanceMeasure(testing_data_y, smote_lr_pred_y).PofB20()
        smote_lr_pofd = PerformanceMeasure(testing_data_y,smote_lr_pred_y).PofD20()
        smote_lr_ranking = PerformanceMeasure(testing_data_y, smote_lr_pred_y).ranking()
        smote_lr_pofb_list.append(smote_lr_pofb)
        smote_lr_pofd_list.append(smote_lr_pofd)

        smote_brr = BayesianRidge().fit(smote_X, smote_y)
        smote_brr_pred_y = smote_brr.predict(testing_data_X)
        smote_brr_fpa = PerformanceMeasure(testing_data_y, smote_brr_pred_y).FPA()
        smote_brr_fpa_list.append(smote_brr_fpa)
        smote_brr_pofb = PerformanceMeasure(testing_data_y, smote_brr_pred_y).PofB20()
        smote_brr_pofd = PerformanceMeasure(testing_data_y,smote_brr_pred_y).PofD20()
        smote_brr_ranking = PerformanceMeasure(testing_data_y, smote_brr_pred_y).ranking()
        smote_brr_pofb_list.append(smote_brr_pofb)
        smote_brr_pofd_list.append(smote_brr_pofd)



if __name__ == '__main__':

    for dataset, filename in Processing().import_single_data():
        csrs_fpa_list = []
        irsvm_fpa_list = []
        svm_fpa_list = []
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
        smote_ltr_fpa_list = []
        smote_rs_fpa_list = []
        smote_dtr_fpa_list = []
        smote_lr_fpa_list = []
        smote_brr_fpa_list = []

        csrs_pofb_list = []
        irsvm_pofb_list = []
        svm_pofb_list = []
        ltr_pofb_list = []
        rs_pofb_list = []
        dtr_pofb_list = []
        lr_pofb_list = []
        brr_pofb_list = []
        rus_ltr_pofb_list = []
        rus_rs_pofb_list = []
        rus_dtr_pofb_list = []
        rus_lr_pofb_list = []
        rus_brr_pofb_list = []
        smote_ltr_pofb_list = []
        smote_rs_pofb_list = []
        smote_dtr_pofb_list = []
        smote_lr_pofb_list = []
        smote_brr_pofb_list = []

        csrs_pofd_list = []
        irsvm_pofd_list = []
        svm_pofd_list = []
        ltr_pofd_list = []
        rs_pofd_list = []
        dtr_pofd_list = []
        lr_pofd_list = []
        brr_pofd_list = []
        rus_ltr_pofd_list = []
        rus_rs_pofd_list = []
        rus_dtr_pofd_list = []
        rus_lr_pofd_list = []
        rus_brr_pofd_list = []
        smote_ltr_pofd_list = []
        smote_rs_pofd_list = []
        smote_dtr_pofd_list = []
        smote_lr_pofd_list = []
        smote_brr_pofd_list = []

        print('filename', filename)
        for _ in range(1):
            bootstrap(dataset=dataset)
            print('bootstrap 迭代一次')

        csrs_fpa_average = sum(csrs_fpa_list) / 5
        irsvm_fpa_average = sum(irsvm_fpa_list) / 5
        svm_fpa_average = sum(svm_fpa_list) / 5
        ltr_fpa_average = sum(ltr_fpa_list) / 5
        rs_fpa_average = sum(rs_fpa_list) / 5
        dtr_fpa_average = sum(dtr_fpa_list) / 5
        lr_fpa_average = sum(lr_fpa_list) / 5
        brr_fpa_average = sum(brr_fpa_list) / 5
        rus_ltr_fpa_average = sum(rus_ltr_fpa_list) / 5
        rus_rs_fpa_average = sum(rus_rs_fpa_list) / 5
        rus_dtr_fpa_average = sum(rus_dtr_fpa_list) / 5
        rus_lr_fpa_average = sum(rus_lr_fpa_list) / 5
        rus_brr_fpa_average = sum(rus_brr_fpa_list) / 5
        smote_ltr_fpa_average = sum(smote_ltr_fpa_list) / 5
        smote_rs_fpa_average = sum(smote_rs_fpa_list) / 5
        smote_dtr_fpa_average = sum(smote_dtr_fpa_list) / 5
        smote_lr_fpa_average = sum(smote_lr_fpa_list) / 5
        smote_brr_fpa_average = sum(smote_brr_fpa_list) / 5

        csrs_pofb_average = sum(csrs_pofb_list) / 5
        irsvm_pofb_average = sum(irsvm_pofb_list) / 5
        svm_pofb_average = sum(svm_pofb_list) / 5
        ltr_pofb_average = sum(ltr_pofb_list) / 5
        rs_pofb_average = sum(rs_pofb_list) / 5
        dtr_pofb_average = sum(dtr_pofb_list) / 5
        lr_pofb_average = sum(lr_pofb_list) / 5
        brr_pofb_average = sum(brr_pofb_list) / 5
        rus_ltr_pofb_average = sum(rus_ltr_pofb_list) / 5
        rus_rs_pofb_average = sum(rus_rs_pofb_list) / 5
        rus_dtr_pofb_average = sum(rus_dtr_pofb_list) / 5
        rus_lr_pofb_average = sum(rus_lr_pofb_list) / 5
        rus_brr_pofb_average = sum(rus_brr_pofb_list) / 5
        smote_ltr_pofb_average = sum(smote_ltr_pofb_list) / 5
        smote_rs_pofb_average = sum(smote_rs_pofb_list) / 5
        smote_dtr_pofb_average = sum(smote_dtr_pofb_list) / 5
        smote_lr_pofb_average = sum(smote_lr_pofb_list) / 5
        smote_brr_pofb_average = sum(smote_brr_pofb_list) / 5

        csrs_pofd_average = sum(csrs_pofd_list) / 5
        irsvm_pofd_average = sum(irsvm_pofd_list) / 5
        svm_pofd_average = sum(svm_pofd_list) / 5
        ltr_pofd_average = sum(ltr_pofd_list) / 5
        rs_pofd_average = sum(rs_pofd_list) / 5
        dtr_pofd_average = sum(dtr_pofd_list) / 5
        lr_pofd_average = sum(lr_pofd_list) / 5
        brr_pofd_average = sum(brr_pofd_list) / 5
        rus_ltr_pofd_average = sum(rus_ltr_pofd_list) / 5
        rus_rs_pofd_average = sum(rus_rs_pofd_list) / 5
        rus_dtr_pofd_average = sum(rus_dtr_pofd_list) / 5
        rus_lr_pofd_average = sum(rus_lr_pofd_list) / 5
        rus_brr_pofd_average = sum(rus_brr_pofd_list) / 5
        smote_ltr_pofd_average = sum(smote_ltr_pofd_list) / 5
        smote_rs_pofd_average = sum(smote_rs_pofd_list) / 5
        smote_dtr_pofd_average = sum(smote_dtr_pofd_list) / 5
        smote_lr_pofd_average = sum(smote_lr_pofd_list) / 5
        smote_brr_pofd_average = sum(smote_brr_pofd_list) / 5

        file = open('result.txt','a')
        file.write(filename)
        file.write("\n")

        file.write('csrs_fpa_list :'+str(csrs_fpa_list))
        file.write("\n")

        file.write('csrs_fap_average: '+str(csrs_fpa_average))
        file.write("\n")

        file.write('irsvm_fpa_average :'+str(irsvm_fpa_average))
        file.write("\n")

        file.write('svm_fpa_average :' + str(svm_fpa_average))
        file.write("\n")

        file.write('ltr_fpa_average :'+str(ltr_fpa_average))
        file.write("\n")

        file.write('rs_fpa_average :'+str(rs_fpa_average))
        file.write("\n")

        file.write('dtr_fpa_average :'+str(dtr_fpa_average))
        file.write("\n")

        file.write('lr_fpa_average :'+str(lr_fpa_average))
        file.write("\n")

        file.write('brr_fpa_average :'+str(brr_fpa_average))
        file.write("\n")

        file.write('rus_ltr_fpa_average :'+str(rus_ltr_fpa_average))
        file.write("\n")

        file.write('rus_rs_fpa_average :'+str(rus_rs_fpa_average))
        file.write("\n")

        file.write('rus_dtr_fpa_average :'+str(rus_dtr_fpa_average))
        file.write("\n")

        file.write('rus_lr_fpa_average :'+str(rus_lr_fpa_average))
        file.write("\n")

        file.write('rus_brr_fpa_average :'+str(rus_brr_fpa_average))
        file.write("\n")

        file.write('smote_ltr_fpa_average :'+str(smote_ltr_fpa_average))
        file.write("\n")

        file.write('smote_rs_fpa_average :'+str(smote_rs_fpa_average))
        file.write("\n")

        file.write('smote_dtr_fpa_average :'+str(smote_dtr_fpa_average))
        file.write("\n")

        file.write('smote_lr_fpa_average :'+str(smote_lr_fpa_average))
        file.write("\n")

        file.write('smote_brr_fpa_average :'+str(smote_brr_fpa_average))
        file.write("\n")


        file.write('csrs_pofb_list :' + str(csrs_pofb_list))
        file.write("\n")

        file.write('csrs_pofb_average: ' + str(csrs_pofb_average))
        file.write("\n")

        file.write('irsvm_pofb_average :' + str(irsvm_pofb_average))
        file.write("\n")

        file.write('svm_pofb_average :' + str(svm_pofb_average))
        file.write("\n")

        file.write('ltr_pofb_average :' + str(ltr_pofb_average))
        file.write("\n")

        file.write('rs_pofb_average :' + str(rs_pofb_average))
        file.write("\n")

        file.write('dtr_pofb_average :' + str(dtr_pofb_average))
        file.write("\n")

        file.write('lr_pofb_average :' + str(lr_pofb_average))
        file.write("\n")

        file.write('brr_pofb_average :' + str(brr_pofb_average))
        file.write("\n")

        file.write('rus_ltr_pofb_average :' + str(rus_ltr_pofb_average))
        file.write("\n")

        file.write('rus_rs_pofb_average :' + str(rus_rs_pofb_average))
        file.write("\n")

        file.write('rus_dtr_pofb_average :' + str(rus_dtr_pofb_average))
        file.write("\n")

        file.write('rus_lr_pofb_average :' + str(rus_lr_pofb_average))
        file.write("\n")

        file.write('rus_brr_pofb_average :' + str(rus_brr_pofb_average))
        file.write("\n")

        file.write('smote_ltr_pofb_average :' + str(smote_ltr_pofb_average))
        file.write("\n")

        file.write('smote_rs_pofb_average :' + str(smote_rs_pofb_average))
        file.write("\n")

        file.write('smote_dtr_pofb_average :' + str(smote_dtr_pofb_average))
        file.write("\n")

        file.write('smote_lr_pofb_average :' + str(smote_lr_pofb_average))
        file.write("\n")

        file.write('smote_brr_pofb_average :' + str(smote_brr_pofb_average))
        file.write("\n")

        file.write('csrs_pofd_list :' + str(csrs_pofd_list))
        file.write("\n")

        file.write('csrs_pofd_average: ' + str(csrs_pofd_average))
        file.write("\n")

        file.write('irsvm_pofd_average :' + str(irsvm_pofd_average))
        file.write("\n")

        file.write('svm_pofd_average :' + str(svm_pofd_average))
        file.write("\n")

        file.write('ltr_pofd_average :' + str(ltr_pofd_average))
        file.write("\n")

        file.write('rs_pofd_average :' + str(rs_pofd_average))
        file.write("\n")

        file.write('dtr_pofd_average :' + str(dtr_pofd_average))
        file.write("\n")

        file.write('lr_pofd_average :' + str(lr_pofd_average))
        file.write("\n")

        file.write('brr_pofd_average :' + str(brr_pofd_average))
        file.write("\n")

        file.write('rus_ltr_pofd_average :' + str(rus_ltr_pofd_average))
        file.write("\n")

        file.write('rus_rs_pofd_average :' + str(rus_rs_pofd_average))
        file.write("\n")

        file.write('rus_dtr_pofd_average :' + str(rus_dtr_pofd_average))
        file.write("\n")

        file.write('rus_lr_pofd_average :' + str(rus_lr_pofd_average))
        file.write("\n")

        file.write('rus_brr_pofd_average :' + str(rus_brr_pofd_average))
        file.write("\n")

        file.write('smote_ltr_pofd_average :' + str(smote_ltr_pofd_average))
        file.write("\n")

        file.write('smote_rs_pofd_average :' + str(smote_rs_pofd_average))
        file.write("\n")

        file.write('smote_dtr_pofd_average :' + str(smote_dtr_pofd_average))
        file.write("\n")

        file.write('smote_lr_pofd_average :' + str(smote_lr_pofd_average))
        file.write("\n")

        file.write('smote_brr_pofd_average :' + str(smote_brr_pofd_average))
        file.write("\n")

        file.close()