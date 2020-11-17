import os
import pandas as pd
import sys
pd.set_option('display.max_columns', None)
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split


def load_data(plane, strain):
    '''
     :param plane: choose plane P123-P27
     :param strain: choose strain 30-35
    '''

    print('Plane ' + plane + ', Strain ' + strain)

    dir = '../data_per_plane/'
    data = pd.read_csv(dir + plane + '_data.csv', encoding='utf-8')
    # M = pd.read_csv(dir+plane+'_mean.csv', encoding='utf-8')
    # S = pd.read_csv(dir+plane+'_sd.csv', encoding='utf-8')

    X = np.asarray(data.get(data.columns.values.tolist()[1:31]))
    y = np.asarray(data.get(strain))

    print('*** data is loaded')

    return X, y


def linear_regression(X, y, fold=5):
    '''
     :method: linear regression
     :param X: independent variable
     :param y: dependent variable
     :param fold: n-fold
    '''

    print('*** Linear Regression model begins')
    MAEs = []
    for i in range(fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        MAE = metrics.mean_absolute_error(y_pred, y_test)
        MAEs.append(MAE)

    print('   The mean absolute error is %f' % (np.mean(MAEs)))
    return np.mean(MAEs)


def ridge_regression(X, y, fold=5):
    '''
     :method: ridge regression
     :param X: independent variable
     :param y: dependent variable
     :param fold: n-fold
    '''

    print('*** Ridge Regression model begins')
    MAEs = []
    for i in range(fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        rr = Ridge()
        rr.fit(X_train, y_train)
        y_pred = rr.predict(X_test)

        MAE = metrics.mean_absolute_error(y_pred, y_test)
        MAEs.append(MAE)

    print('   The mean absolute error is %f' % (np.mean(MAEs)))
    return np.mean(MAEs)


def lasso_regression(X, y, fold=5):
    '''
     :method: lasso regression
     :param X: independent variable
     :param y: dependent variable
     :param fold: n-fold
    '''

    print('*** Lasso Regression model begins')
    MAEs = []
    for i in range(fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        lr = Lasso()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)

        MAE = metrics.mean_absolute_error(y_pred, y_test)
        MAEs.append(MAE)

    print('   The mean absolute error is %f' % (np.mean(MAEs)))
    return np.mean(MAEs)


def bayesian_ridge_regression(X, y, fold=5):
    '''
     :method: bayesian ridge regression
     :param X: independent variable
     :param y: dependent variable
     :param fold: n-fold
    '''

    print('*** Bayesian Ridge Regression model begins')
    MAEs = []
    for i in range(fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        br = BayesianRidge()
        br.fit(X_train, y_train)
        y_pred = br.predict(X_test)

        MAE = metrics.mean_absolute_error(y_pred, y_test)
        MAEs.append(MAE)

    print('   The mean absolute error is %f' % (np.mean(MAEs)))
    return np.mean(MAEs)


def MLP(X, y, fold=5, dim=64):
    '''
     :method: multi-layer perceptron
     :param X: independent variable
     :param y: dependent variable
     :param fold: n-fold
    '''

    print('*** Multi-Layer Perceptron model begins')
    MAEs = []
    for i in range(fold):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        mlp = MLPRegressor(hidden_layer_sizes=dim, activation='logistic', solver='adam', learning_rate='adaptive',
                           early_stopping=True, random_state=1, max_iter=500)
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)

        MAE = metrics.mean_absolute_error(y_pred, y_test)
        MAEs.append(MAE)

    print('   The mean absolute error is %f' % (np.mean(MAEs)))
    return np.mean(MAEs)


def main(plane, strain, model, fold, dim):
    '''
     :param model: choose model
     :param strain: choose strain 30-35
    '''

    # load data
    X, y = load_data(plane, strain)

    # choose model
    if model == 'linear regression':
        result = linear_regression(X, y, fold)

    if model == 'ridge regression':
        result = ridge_regression(X, y, fold)

    if model == 'lasso regression':
        result = lasso_regression(X, y, fold)

    if model == 'bayesian ridge regression':
        result = bayesian_ridge_regression(X, y, fold)

    if model == 'MLP':
        result = MLP(X, y, fold, dim)

    return result


if __name__ == "__main__":

    # main(plane='P123', strain='30', model='MLP', fold=1, dim=64)

    plane_list = ['P123', 'P124', 'P125', 'P126', 'P127']
    strain_list = ['30', '31', '32', '33', '34', '35']
    model_list = ['linear regression', 'ridge regression', 'lasso regression', 'bayesian ridge regression', 'MLP']

    results = []
    for model in model_list:
        for plane in plane_list:
            for strain in strain_list:
                result = main(plane, strain, model, fold=1, dim=64)
                results.append(result)

    with open('../result/results_basic_regression_model.pkl', 'wb') as file:
        pickle.dump(results, file)
    print('[******]all methods done!')
