#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic
from logic.analyze.sklearn.enum.e_sklearn_type import ESklearnType
from .sklearn_logic_classification import SklearnLogicClassification
from .sklearn_logic_regression import SklearnLogicRegression

sys.path.append('/../../')

class SklearnLogic(AbstractLogic):

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        if self._input.X_train is None or self._input.Y_train is None:
            raise Exception('X_train or Y_train Nothing')

        self._output.model_name = self._input.model.__class__.__name__
        self._logger.debug("モデル:{}".format(self._output.model_name) + " ***********************")

        grid_search = copy.deepcopy(self._input.model)
        random_search = copy.deepcopy(self._input.model)

        results: list = []

        # トレーニングデータ、テストデータの分離
        train_X, test_X, train_y, test_y = train_test_split(
            self._input.X_train, self._input.Y_train, random_state=0)

        # グリッドサーチ
        if len(self._input.grid_search_params) > 0:
            clf = GridSearchCV(grid_search, self._input.grid_search_params, cv = self._input.cv_value)
            clf.fit(train_X, train_y)
            pred_y = clf.predict(test_X)
            score = f1_score(test_y, pred_y, average="micro")

            self._logger.debug('サーチ方法:グリッドサーチ =======================')
            self._logger.debug("ベストスコア:{}".format(score))
            self._logger.debug("パラメーター:{}".format(clf.best_params_))
            results.append({
                'Type': 'GridSearch',
                'Score': score,
                'Params': clf.best_params_,
                'Target_cols': self._input.X_train.columns,
                'Pred_y': pred_y
            })

        # ランダムサーチ
        if len(self._input.random_search_params) > 0:
            clf = RandomizedSearchCV(random_search, self._input.random_search_params, cv = self._input.cv_value)
            clf.fit(train_X, train_y)
            pred_y = clf.predict(test_X)
            score = f1_score(test_y, pred_y, average="micro")

            self._logger.debug('サーチ方法:ランダムサーチ =======================')
            self._logger.debug("ベストスコア:{}".format(score))
            self._logger.debug("パラメーター:{}".format(clf.best_params_))
            results.append({
                'Type': 'RandomSearch',
                'Score': score,
                'Params': clf.best_params_,
                'Target_cols': self._input.X_train.columns,
                'Pred_y': pred_y
            })

        # デフォルトサーチ
        model = copy.deepcopy(self._input.model)

        ret = False
        if self._input.sklearn_type == ESklearnType.CLASSIFICATION:
            classification = SklearnLogicClassification(self._input, self._output)
            ret = classification.fit(model, self._input.X_train, self._input.Y_train, test_X, test_y)
            self._output = classification._output
        else:
            regression = SklearnLogicRegression(self._input, self._output)
            ret = regression.fit(model, train_X, train_y)
            self._output = regression._output

        return ret
