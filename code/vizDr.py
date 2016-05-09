#!/usr/bin/env python
# vizDr.py
#
#
# Title:        Visualization tools for machine learning
# Version:      1.0
# Author:       Rebecca Bilbro
# Date:         5/9/16
# Organization: District Data Labs

#############################################################################
# Imports
#############################################################################
import os
import zipfile
import requests
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import radviz
from pandas.tools.plotting import parallel_coordinates

from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize

from sklearn import cross_validation as cv

from sklearn.metrics import auc
from sklearn.metrics import r2_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error as mse

from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import validation_curve

from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsClassifier


#############################################################################
# Get the Data
#############################################################################

sns.set_style("whitegrid")

OCCUPANCY = "http://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"
CREDIT    = "http://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
CONCRETE  = "http://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"

def download_data(url, path='data'):
    if not os.path.exists(path):
        os.mkdir(path)

    response = requests.get(url)
    name = os.path.basename(url)
    with open(os.path.join(path, name), 'w') as f:
        f.write(response.content)


#############################################################################
# Visual feature analysis tools
#############################################################################

def box_viz(df):
    ax = sns.boxplot(df)
    plt.xticks(rotation=60)
    plt.show()

def hist_viz(df,feature):
    ax = sns.distplot(df[feature])
    plt.xlabel(feature)
    plt.show()

def splom_viz(df,labels=None):
    ax = sns.pairplot(df, hue=labels, diag_kind="kde", size=2)
    plt.show()

def pcoord_viz(df, labels):
    fig = parallel_coordinates(df, labels, color=sns.color_palette())
    plt.show()

def rad_viz(df,labels):
    fig = radviz(df, labels, color=sns.color_palette())
    plt.show()

def joint_viz(feat1,feat2,df):
    ax = sns.jointplot(feat1, feat2, data=df, kind='reg', size=5)
    plt.xticks(rotation=60)
    plt.show()

#############################################################################
# Visual model evaluation tools
#############################################################################

def plot_classification_report(cr, title='Classification report', cmap=plt.cm.Reds):
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center')

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()

def rocViz(y, yhat, model):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y,yhat)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.title('Receiver Operating Characteristic for %s' % model)
    plt.plot(false_positive_rate, true_positive_rate, 'blue', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'m--')
    plt.xlim([0,1])
    plt.ylim([0,1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def regrErrorViz(model,features,labels):
    predicted = cv.cross_val_predict(model, features, labels, cv=12)
    fig, ax = plt.subplots()
    ax.scatter(labels, predicted)
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()

def plotResids(features,labels,model):
    for feature in list(features):
        splits = cv.train_test_split(features[[feature]], labels, test_size=0.2)
        X_train, X_test, y_train, y_test = splits
        model.fit(X_train, y_train)
        plt.scatter(model.predict(X_train), model.predict(X_train) - y_train, c='b', s=40, alpha=0.5)
        plt.scatter(model.predict(X_test), model.predict(X_test) - y_test, c='g', s=40)
    plt.hlines(y=0, xmin=0, xmax=100)
    plt.title('Plotting residuals using training (blue) and test (green) data')
    plt.ylabel('Residuals')
    plt.xlim([20,70])
    plt.ylim([-50,50])
    plt.show()



if __name__ == '__main__':
    ## You only need to do this once!!
    # download_data(OCCUPANCY)
    # download_data(CREDIT)
    # download_data(CONCRETE)
    # z = zipfile.ZipFile(os.path.join('data', 'occupancy_data.zip'))
    # z.extractall(os.path.join('data', 'occupancy_data'))

    occupancy   = pd.read_csv(os.path.join('data','occupancy_data','datatraining.txt'), sep=",")
    credit      = pd.read_excel(os.path.join('data','default%20of%20credit%20card%20clients.xls'), header=1)
    concrete    = pd.read_excel(os.path.join('data','Concrete_Data.xls'))
    occupancy.columns = ['date', 'temp', 'humid', 'light', 'co2', 'hratio', 'occupied']
    concrete.columns = ['cement', 'slag', 'ash', 'water', 'splast', 'coarse', 'fine','age','strength']
    credit.columns = ['id','limit','sex','edu','married','age','apr_delay','may_delay','jun_delay','jul_delay',\
                      'aug_delay','sep_delay','apr_bill','may_bill','jun_bill','jul_bill','aug_bill','sep_bill',\
                      'apr_pay','may_pay','jun_pay','jul_pay','aug_pay','sep_pay','default']
