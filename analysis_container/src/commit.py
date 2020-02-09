
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


try:
    print('start')
    execute()

except Exception as e:
    print('error')
    print(e)
else:
    print('success')
finally:
    print('end')
