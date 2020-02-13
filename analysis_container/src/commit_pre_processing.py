#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import copy
import time
import math
import pandas as pd

from container.simulator_container import SimulatorContainer
from container.logic_dict import LogicDict
from logic.analyze.sklearn.interface.i_sklearn_input import ISklearnInput
from logic.data_wrangling.pre_processing.interface.i_pre_processing_input import IPreProcessingInput
from logic.analyze.light_gbm.interface.i_light_gbm_input import ILightGbmInput

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import scipy.stats

import lightgbm as lgb

def execute():
    start = time.time()

    train_df = pd.read_pickle("data/output/train_df.pkl")
    test_df = pd.read_pickle("data/output/test_df.pkl")

    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test = test_df.drop("PassengerId", axis=1).copy()

    # target_cols: list = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Title', 'FamilySize']
    target_cols: list = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
    drop_cols = []
    for col in X_train.columns:
        if not col in target_cols:
            drop_cols.append(col)

    print("X_train.head()")
    print(X_train.head())
    print("X_test.head()")
    print(X_test.head())

    if len(drop_cols) > 0:
        X_train = X_train.drop(drop_cols, axis=1)
        X_test = X_test.drop(drop_cols, axis=1)

    # model = RandomForestClassifier(n_estimators=100)
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    model.fit(X_train, Y_train)
    pred_y = model.predict(X_test)
    # score = model.score(X_test, pred_y)

    print("X_train count:{0}".format(len(X_train)))
    print("Y_train count:{0}".format(len(Y_train)))
    print("X_test  count:{0}".format(len(X_test)))
    print("pred_y  count:{0}".format(len(pred_y)))
    # print("score        :{0}".format(str(score)))

    submission = pd.DataFrame({
            "PassengerId": test_df["PassengerId"],
            "Survived": pred_y
        })
    submission.to_csv('data/output/submission.csv', index=False)

    elapsed_time = math.floor(time.time() - start)
    print ("total elapsed_time :{0}".format(elapsed_time) + "[sec]")

def output_best_features():
    start = time.time()

    # 新規にデータ事前処理をする場合はコメントアウト
    df_score = pd.read_pickle("data/output/df_score.pkl")
    df_score_s = df_score.sort_values('score', ascending=False)
    print(df_score_s)
    print(df_score_s[['model', 'score', 'target_cols']].values[0:20])

    # RandomForestClassifier 0.860987〜 TOP 5
    # [('Pclass', 'Age', 'Parch', 'Fare', 'Embarked', 'FamilySize')
    # ('Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'FamilySize')
    # ('Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'FamilySize')
    # ('Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'FamilySize')
    # ('Pclass', 'Sex', 'Age', 'Parch', 'Fare', 'Embarked', 'Title')]

    elapsed_time = math.floor(time.time() - start)
    print ("total elapsed_time :{0}".format(elapsed_time) + "[sec]")

def get_sklearn_models() -> list:
    return [
        {
            # ロジスティック回帰(分類)
            'model_name': 'LogisticRegression',
            'model_instance': LogisticRegression(solver='liblinear'),
            'grid_search_params': {
                "C": [10 ** i for i in range(-5, 6)],
                "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
                "C": scipy.stats.uniform(0.00001, 1000),
                "random_state": scipy.stats.randint(0, 100)
            }
        },
        {
            # 線形SVC(分類)
            'model_name': 'LinearSVC',
            'model_instance': LinearSVC(dual=False),
            'grid_search_params': {
                "C": [10 ** i for i in range(-5, 6)],
                # "multi_class": ["ovr", "crammer_singer"],
                "class_weight": ["balanced"],
                # "random_state": [i for i in range(0, 101)]
                "random_state": [100]
            },
            'random_search_params': {
                "C": scipy.stats.uniform(0.00001, 1000),
                # "multi_class": ["ovr", "crammer_singer"],
                "class_weight": ["balanced"],
                # "random_state": scipy.stats.randint(0, 100)
                "random_state": [100]
            }
        },
        {
            # 非線形SVC(分類)
            'model_name': 'SVC',
            'model_instance': SVC(gamma="scale"),
            'grid_search_params': {
                "C": [10 ** i for i in range(-5, 6)],
                # "kernel": ["linear", "rbf", "sigmoid"],
                # "decision_function_shape": ["ovo", "ovr"],
                # "random_state": [i for i in range(0, 101)]
                "random_state": [100]
            },
            'random_search_params': {
                "C": scipy.stats.uniform(0.00001, 1000),
                # "kernel": ["linear", "rbf", "sigmoid"],
                # "decision_function_shape": ["ovo", "ovr"],
                # "random_state": scipy.stats.randint(0, 100)
                "random_state": [100]
            }
        },
        {
            # k近傍法
            'model_name': 'KNeighborsClassifier',
            'model_instance': KNeighborsClassifier(n_neighbors=3),
            'grid_search_params': {
                'weights': ['uniform', 'distance'],
                'n_neighbors': [3, 5, 10, 20, 30, 50]
            },
            'random_search_params': {
                'weights': ['uniform', 'distance'],
                'n_neighbors': [3, 5, 10, 20, 30, 50]
            }
        },
        {
            # Gaussian Naive Bayes
            'model_name': 'GaussianNB',
            'model_instance': GaussianNB(),
            'grid_search_params': {},
            'random_search_params': {}
        },
        {
            # Perceptron
            'model_name': 'Perceptron',
            'model_instance': Perceptron(),
            'grid_search_params': {
                # 'alpha': [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                # 'n_iter': [5, 10, 15, 20, 50]
            },
            'random_search_params': {}
        },
        {
            # DecisionTreeClassifier
            'model_name': 'DecisionTreeClassifier',
            'model_instance': DecisionTreeClassifier(),
            'grid_search_params': {
                # "criterion": ["gini", "entropy"],
                # "splitter": ["best", "random"],
                # "max_depth": [i for i in range(1, 11)],
                # "min_samples_split": [i for i in range(2, 11)],
                # "min_samples_leaf": [i for i in range(1, 11)],
                # "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
            }
        },
        {
            # RandomForestClassifier
            'model_name': 'RandomForestClassifier',
            'model_instance': RandomForestClassifier(n_estimators=100),
            'grid_search_params': {
                # "criterion": ["gini", "entropy"],
                # "splitter": ["best", "random"],
                # "max_depth": [i for i in range(1, 11)],
                # "min_samples_split": [i for i in range(2, 11)],
                # "min_samples_leaf": [i for i in range(1, 11)],
                # "random_state": [i for i in range(0, 101)]
            },
            'random_search_params': {
            }
        },
    ]    

def get_list_all_pattern_count(patterns: list, limit_dimensionality_num: int = 2):
    all_pattern = [copy.deepcopy(patterns)]
    for i in range(len(patterns)):
        copy_patterns = copy.deepcopy(patterns)
        copy_patterns.pop(i)
        all_pattern.append(copy_patterns)

        if len(copy_patterns) > limit_dimensionality_num:
            all_pattern.extend  (get_list_all_pattern_count(copy_patterns, limit_dimensionality_num))
    
    return set(set(list(map(tuple,all_pattern))))

try:
    print('start')
    # execute()

    output_best_features()

    # all_pattern: list = get_list_all_pattern_count(
    #     ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize'],
    #     6)
    # print(all_pattern)
    # print(len(all_pattern))

except Exception as e:
    print('error')
    print(e)
else:
    print('success')
finally:
    print('end')
