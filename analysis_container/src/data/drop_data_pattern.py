# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns

# machine learning
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

import lightgbm as lgb


def execute():
    # 基準となる列drop前のデータを読み込む
    train_df = pd.read_pickle("output/train_df.pkl")
    test_df = pd.read_pickle("output/test_df.pkl")
    combine = [train_df, test_df]

    # データ削除のパターンを返す

def get_col_drop_list():
    # 削除対象列は以下
    # Name, Ticket, Cabin, AgeBand, FareBand

    # 全削除

    # 1col残し

    # 2col残し

    # 2col残し

    return True


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
