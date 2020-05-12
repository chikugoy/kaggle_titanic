
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

try:
    print('start')
    output_best_features()

except Exception as e:
    print('error')
    print(e)
else:
    print('success')
finally:
    print('end')
