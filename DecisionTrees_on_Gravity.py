#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gravity random forest, ensembles
OOP

@author: adrienne bohlmann

python 3.7.7
numpy 1.19.2
scikit-learn 0.24.1
tensorflow 2.0.0    
keras 2.3.1
matplitlib 3.3.2

"""
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

import matplotlib.pyplot as plt

##############################################################################
# fix randomness in train-test-split 
# nRtts = 42
# or keep randomness to observe robustness?
nRtts = None

##############################################################################
# load, select and first preprocessing of data
#  
# zipped data and explatations obtained from 
# https://sites.google.com/site/hiegravity/data-sources [29.03.2020]
# "lighter version" 

#load data from working directory
gravdata = pd.read_stata('col_regfile09.dta')

# have a first look 
gravdata.columns

gravdata.year.nunique()
gravdata.year.unique

# select a single year for cross section analysis (no time series)
gravdata = gravdata[gravdata.year.isin([2000])]

gravdata.isna().sum()

# select variables of potential interest 
gravdata = gravdata[['flow', 'distw'
                     , 'gdp_o', 'gdp_d'
                     , 'contig', 'comlang_off', 'col_hist']]

# observe and then delete missing observations
gravdata.isna().sum()
gravdata = gravdata.dropna()

# look at the variables
gravdata.dtypes
gravdata.describe()

# corr between explanatory variables must not be too high
correlation = gravdata.corr()
# at first sight low correlation among all variables


# trade flow from origin to destination
flow = np.array(gravdata.flow.astype(np.float32))
# zero trade
print('of', len(flow), 'observations'
      , len(flow) - np.count_nonzero(flow), ' are zero ='
      , (len(flow) - np.count_nonzero(flow))/ len(flow)*100, '%')

# create a binary variable indicating Zero trade for stratify in train-test-split
flowZ = flow.copy()
flowZ[flowZ > 0] = 1

# reformat

# weighted geographical between origin and destination (see documentation for details)
distw = np.array(gravdata.distw.astype(np.float32))

# GDP of origin
gdp_o = np.array(gravdata.gdp_o.astype(np.float32))

# GDP of destination
gdp_d = np.array(gravdata.gdp_d.astype(np.float32))

# Dummy indicating neighbourhood status between origin and destination
contig = np.array(gravdata.contig.astype(np.int8))

# Dummy indicating common official language of origin and destination
comlang_off = np.array(gravdata.comlang_off.astype(np.int8))

# Dummy indicating colonial ties between origin and destination
col_hist = np.array(gravdata.col_hist.astype(np.int8))

# look at histograms of the continuus variables
plt.hist(flow, bins=1000, log=True)
plt.title('hist log trade flow')
plt.show()

plt.hist(distw, bins=1000)
plt.title('hist weighted distance')
plt.show()

plt.hist(gdp_o, bins=1000, log=True)
plt.title('hist log GDP')
plt.show()

# format features

X = np.array([distw, gdp_o, gdp_d
              , contig
              , comlang_off
              , col_hist
              , flowZ
              ]).T


##############################################################################
# class and functions

# class for data preparation
class prepare_data:
    def __init__(self):
        self.X = X.copy()
        self.y = flow.copy()
        
    # train test split and transform
    def tts(self, rnd = 42):
        # train test split keeping only n_X explanatory variables
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X[:,0:6], self.y
                                                                                , random_state=(rnd)
                                                                                , stratify=(self.X[:,6])
                                                                                )
   
    def plot_hist(self):
        # unscaled data
        plt.hist(self.X_train[:, 0], alpha=0.6, bins=100, color='brown')
        plt.hist(self.X_train[:, 1], alpha=0.6, bins=100, color='orange')
        plt.title('selected unscaled training data')
        plt.legend(['weighted distance', 'GDP'])
        plt.show()
        
        # scaled data
        plt.hist(self.X_train_scaled[:, 0], alpha=0.6, bins=100, color='brown')
        plt.hist(self.X_train_scaled[:, 1], alpha=0.6, bins=100, color='orange')
        # hist_title = str(self.scaler) + ' training data'
        plt.title('selected scaled training data')
        plt.legend(['weighted distance', 'GDP'])
        plt.show()
        

def plot_feature_importance(model):
    n_features = len(data.X_train[0,:])
    plt.barh(range(n_features), model.feature_importances_, align='center') 
    plt.yticks(np.arange(n_features), feature_names) 
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

def cross_val(estimator, cv=21):
    all_r2_test = []
    for i in range(cv+1):
        data.tts(rnd=None)
        estimator.fit(data.X_train, data.y_train)
        r2_test = estimator.score(data.X_test, data.y_test)
        all_r2_test.append(r2_test)
        print('R2 test:', r2_test)
        print('R2 train', estimator.score(data.X_train, data.y_train))
    print('mean R2 test:', np.mean(all_r2_test))


def get_R2s(estimator):
    print('test R2:', estimator.score(data.X_test, data.y_test))
    print('train R2:', estimator.score(data.X_train, data.y_train))
    

def plt_y_pred(estimator):
    y_pred = estimator.predict(data.X_test)
    # plot feature importance
    plot_feature_importance(estimator)
    plt.show()
    # plot y against y^
    plt.scatter(y_pred, data.y_test, s = 3)
    plt.xlabel('prediced trade flow from test data')
    plt.ylabel('true trade flow from test data')
    plt.title('test vs predicted trade flow')
    plt.show()
    # plot y against y^ on log axes
    plt.loglog(y_pred, data.y_test, marker = 'o', markersize = 3, linestyle = '')
    plt.xlabel('prediced trade flow from test data')
    plt.ylabel('true trade flow from test data')
    plt.title('test vs predicted trade flow on log axes')
    plt.show()
    
    
##############################################################################
# prepare data

data = prepare_data()

# reproducible data split
# data.tts(rnd = 42)
# random data split
data.tts(rnd = None)

# have a look
data.plot_hist()

# retrieve feature names
feature_names = gravdata.columns[1:len(X[0,:])]

##############################################################################
# tree

tree = DecisionTreeRegressor(random_state=42, max_depth = 7) 
tree.fit(data.X_train, data.y_train)

get_R2s(tree)
plt_y_pred(tree)

# plot the tree
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot', feature_names=feature_names, filled=True)

import graphviz
with open('tree.dot') as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

cross_val(tree)


##############################################################################
# grid search for one tree

# criterion
crit = ['mse', 'friedman_mse', 'poisson']

# max depth
maxD = [2, 3, 5, 8, 13, 21, 34]

# create the grid
grid = {'criterion': crit, 'max_depth': maxD}

# base model to tune
baseTree = DecisionTreeRegressor()

# random grid search for parameters
tree_grid_search = GridSearchCV(estimator = baseTree
                                , param_grid= grid
                                , cv = 21              # no of cross validation
                                , n_jobs = -1         # use all available cores
                                )

for i in range(21):
    # if grid is to run on a different data split, run this:
    data.tts(rnd = None)
    # fixed data split:
    # data.tts(rnd = 42)
    # single grid search:
    tree_grid_search.fit(data.X_train, data.y_train)
    print(tree_grid_search.best_params_)
    
# frequent results
# 'friedman_mse'



##############################################################################
# forest

forest = RandomForestRegressor(random_state=42) 
forest.fit(data.X_train, data.y_train)

get_R2s(forest)
plt_y_pred(forest)

##############################################################################
# grid search forest

# no. of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 500, num = 50)]
# max levels in tree
max_depth = [int(x) for x in np.linspace(5, 100, num = 5)]
max_depth.append(None)

# create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               }

# base model to tune
baseForest = RandomForestRegressor()
# random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = baseForest
                               , param_distributions = random_grid
                               , n_iter = 100, cv = 3, verbose=2
                               #, random_state=42
                               , n_jobs = -1)

# if grid is to run on a different data split, run this:
data.tts(rnd = None)
# fixed data split:
#data.tts(rnd = 42)

# single grid search:
rf_random.fit(data.X_train, data.y_train)
rf_random.best_params_

# rnd = 42 gives 
# {'n_estimators': 270, 'max_depth': 76}
# {'n_estimators': 86, 'max_depth': 28}
# {'n_estimators': 77, 'max_depth': 52}
# random tts
# {'n_estimators': 50, 'max_depth': 52}
# {'n_estimators': 68, 'max_depth': 52}
# {'n_estimators': 50, 'max_depth': 52}


##############################################################################
# forest trying to use approx. grid search results 

forest_opt = RandomForestRegressor(n_estimators=200, max_depth=52
                               # random_state=42, 
                               ) 

forest_opt.fit(data.X_train, data.y_train)

get_R2s(forest_opt)
plt_y_pred(forest_opt)

cross_val(forest_opt)


##############################################################################
# cross validation
# two sources of randomness: in train test split and the forest

# first, test only random train test split, holding the forest fixed
forest_opt2 = RandomForestRegressor(n_estimators=377, max_depth=13)

cross_val(forest_opt2)

##############################################################################
# gradient boosting
# with cross validation

booster = GradientBoostingRegressor()

cross_val(booster)
    
##############################################################################
# grid search for boosting

# loss
loss = ['ls', 'lad', 'huber', 'quantile']
# loss = ['ls', 'huber', 'quantile']
# learning rate
lr = [0.001, 0.01, 0.1, 1]
# subsample below 1 leads to stochastic gradient boosting 
subs = [0.5, 0.8, 1.0]
# max depth
maxD = [2, 3, 5]

# create the random grid
random_grid = {'loss': loss
               , 'learning_rate': lr
               #, 'subsample': subs
               , 'max_depth': maxD
               }

# base model to tune
baseBoost = GradientBoostingRegressor()

# random grid search for parameters
boost_random = RandomizedSearchCV(estimator = baseBoost
                                  , param_distributions = random_grid
                                  , n_iter =30           # different combinations to search
                                  , cv = 3              # 3 fold cross validation
                                  # , verbose=2
                                  #, random_state=42
                                  , n_jobs = -1         # use all available cores
                                  )


# if grid is to run on a different data split, run this:
# data.tts(rnd = None)
# fixed data split:
# data.tts(rnd = 42)

# single grid search:
boost_random.fit(data.X_train, data.y_train)
boost_random.best_params_

# results: 
# {'max_depth': 3, 'loss': 'ls', 'learning_rate': 0.1}
# {'max_depth': 8, 'loss': 'huber', 'learning_rate': 0.1}


##############################################################################
# booster with above results

booster_opt = GradientBoostingRegressor(learning_rate=0.1   
                                        , loss = 'huber'          # 'huber' or 'ls' 
                                        , max_depth = 8           # 8 or 3
                                        , n_estimators = 200
                                        )

# cross validating with random tts
cross_val(booster_opt)

# stochastic gradient descent  boosting
booster_SGD = GradientBoostingRegressor(subsample = 0.8
                                        , max_depth=5       
                                        , n_estimators = 200
                                        )
        
cross_val(booster_SGD)        

# have a look?
get_R2s(booster_SGD)
plt_y_pred(booster_SGD)

##############################################################################
# AdaBoost

# define regressor that is to be boosted within AdaBoost
# from the grid search above: 
# 'friedman_mse', 'max_depth': ??
                        
ada = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(criterion = 'friedman_mse', max_depth=34))

ada.fit(data.X_train, data.y_train)

get_R2s(ada)

plt_y_pred(ada)

cross_val(ada)
















    
    
    
    
    
