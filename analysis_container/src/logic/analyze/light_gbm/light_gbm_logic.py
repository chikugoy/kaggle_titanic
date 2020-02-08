#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import scipy.stats
import random as rnd
from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic
import sys

sys.path.append('/../../')


class LightGbmLogic(AbstractLogic):

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        if self._input.X_train is None or self._input.Y_train is None:
            raise Exception('X_train or Y_train Nothing')

        results: list = []

        # LightGBMの分類器をインスタンス化
        gbm = lgb.LGBMClassifier(objective='binary')

        self._output.model_name = gbm.__class__.__name__
        self._logger.debug("モデル:{}".format(
            self._output.model_name) + " ***********************")

        # 試行するパラメータを羅列する
        params = {
            'max_depth': [2, 3, 4, 5],
            'reg_alpha': [0, 1, 10, 100],
            'reg_lambda': [0, 1, 10, 100],
        }

        grid_search = GridSearchCV(
            gbm,  # 分類器を渡す
            param_grid=params,  # 試行してほしいパラメータを渡す
            cv=3,  # 3分割交差検証でスコアを確認
        )

        grid_search.fit(self._input.X_train, self._input.Y_train)  # データを渡す

        self._logger.debug('サーチ方法:グリッドサーチ =======================')
        self._logger.debug("ベストスコア:{}".format(grid_search.best_score_))
        self._logger.debug("パラメーター:{}".format(grid_search.best_params_))
        results.append({
            'Type': 'GridSearch',
            'Score': grid_search.best_score_,
            'Params': grid_search.best_params_,
            'Target_cols': self._input.X_train.columns
        })

        # X_trainとY_trainをtrainとvalidに分割
        train_x, valid_x, train_y, valid_y = train_test_split(
            self._input.X_train, self._input.Y_train, test_size=0.33, random_state=0)

        # trainとvalidを指定し学習
        gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                early_stopping_rounds=20,  # 20回連続でlossが下がらなかったら終了
                verbose=10  # 10round毎に、lossを表示
                )

        # valid_xについて推論
        # oofはout of fold
        oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)
        print('score', round(accuracy_score(valid_y, oof)*100, 2), '%')  # 正解率の表示

        # # testの予測
        # test_pred = gbm.predict(test, num_iteration=gbm.best_iteration_)  # testの予測
        # sample_submission['Survived'] = test_pred  # sample_submissionのSurvived列をtest_predに置き換え
        # sample_submission.to_csv('train_test_split.csv', index=False)  # csvファイルの書き出し

        kf = KFold(n_splits=3)  # 3分割交差検証のためにインスタンス化

        # スコアとモデルを格納するリスト
        score_list = []
        models = []

        print(self._input.X_train.columns)

        # 重要度
        print(pd.DataFrame({'特徴': self._input.X_train.columns,
                            'importance': gbm.feature_importances_}).sort_values('importance',
                                                                                 ascending=False))

        for fold_, (train_index, valid_index) in enumerate(kf.split(self._input.X_train, self._input.Y_train)):
            train_x = self._input.X_train.iloc[train_index]
            valid_x = self._input.X_train.iloc[valid_index]
            train_y = self._input.Y_train[train_index]
            valid_y = self._input.Y_train[valid_index]

            print(f'fold{fold_ + 1} start')

            # {'max_depth': 2, 'reg_alpha': 0, 'reg_lambda': 10}
            gbm = lgb.LGBMClassifier(objective='binary', max_depth=2, reg_alpha=0,
                                     reg_lambda=10, importance_type='gain')
            gbm.fit(train_x, train_y, eval_set=[(valid_x, valid_y)],
                    early_stopping_rounds=20,
                    categorical_feature=['Sex', 'Embarked', 'Age'],
                    verbose=-1)  # 学習の状況を表示しない

            oof = gbm.predict(valid_x, num_iteration=gbm.best_iteration_)
            score_list.append(round(accuracy_score(valid_y, oof)*100, 2))
            models.append(gbm)  # 学習が終わったモデルをリストに入れておく
            print(f'fold{fold_ + 1} end\n')

        print(score_list, '平均score', np.mean(score_list), "%")

        self._logger.debug('サーチ方法:デフォルトサーチ =======================')
        self._logger.debug("スコア:" + str(np.mean(score_list)))
        results.append({
            'Type': 'Default',
            'Score': np.mean(score_list) / 100,
            'Params': None,
            'Target_cols': self._input.X_train.columns
        })

        self._output.results = results

        return True
