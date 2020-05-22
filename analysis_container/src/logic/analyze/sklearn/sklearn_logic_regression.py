#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
import scipy.stats
import random as rnd

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from logic.abstract_interface import AbstractInterface
from logic.analyze.sklearn.enum.e_sklearn_type import ESklearnType 
from logic.abstract_logic import AbstractLogic

sys.path.append('/../../')

class SklearnLogicRegression(AbstractLogic):
    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        return True

    def fit(self, model, train_X, train_y) -> bool:
        kfold = KFold(n_splits=5, random_state=42)
        pred_y = cross_val_predict(model, train_X, train_y, cv = kfold)
        score = r2_score(train_y, pred_y)
        rmse = np.sqrt(mean_squared_error(train_y, pred_y))
        mae = mean_absolute_error(train_y, pred_y)
        rmse_mae = rmse / mae
        self.create_yyplot(train_y, pred_y)

        self._logger.debug('サーチ方法:デフォルトサーチ(回帰:' + self._output.model_name + ') =======================')
        self._logger.debug("SCORE:" + str(score))
        self._logger.debug("RMSE:" + str(rmse))
        self._logger.debug("MAE:" + str(mae))
        self._logger.debug("RMSE / MAE:" + str(rmse_mae))
        results: list = []
        results.append({
            'Type': 'Default',
            'Score': score,
            'Rmse': rmse,
            'Mae': mae,
            'Rmse_mae': rmse_mae,
            'Params': None,
            'Target_cols': self._input.X_train.columns,
            'Pred_y': pred_y
        })

        self._output.results = results

        return True

    def create_yyplot(self, y_obs, y_pred):
        yvalues = np.concatenate([y_obs.flatten(), y_pred.flatten()])
        ymin, ymax, yrange = np.amin(yvalues), np.amax(yvalues), np.ptp(yvalues)
        plt.figure(figsize=(8, 8))
        plt.scatter(y_obs, y_pred)
        plt.plot([ymin - yrange * 0.01, ymax + yrange * 0.01], [ymin - yrange * 0.01, ymax + yrange * 0.01])
        plt.xlim(ymin - yrange * 0.01, ymax + yrange * 0.01)
        plt.ylim(ymin - yrange * 0.01, ymax + yrange * 0.01)
        plt.xlabel('y_observed', fontsize=24)
        plt.ylabel('y_predicted', fontsize=24)
        plt.title('Observed-Predicted Plot', fontsize=24)
        plt.tick_params(labelsize=16)
        plt.savefig('image/output/regression_yyplot.png')
