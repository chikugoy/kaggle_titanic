#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from logic.abstract_interface import AbstractInterface
from logic.analyze.sklearn.enum.e_sklearn_type import ESklearnType
from logic.abstract_logic import AbstractLogic

sys.path.append('/../../')

class SklearnLogicClassification(AbstractLogic):
    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        return True

    def fit(self, model, train_X, train_y, test_X, test_y) -> bool:
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)

        # score = round(model.score(train_X, train_y) * 100, 2) / 100
        score           = round(accuracy_score(test_y, pred_y) * 100, 2) / 100
        precision       = round(precision_score(test_y, pred_y) * 100, 2) / 100
        recall          = round(recall_score(test_y, pred_y) * 100, 2) / 100
        f1              = round(f1_score(test_y, pred_y) * 100, 2) / 100
        classification  = classification_report(test_y, pred_y, output_dict=True)

        self._logger.debug('サーチ方法:デフォルトサーチ(分類:' + self._output.model_name + ') =======================')
        self._logger.debug("スコア:" + str(score))
        results: list = []
        results.append({
            'Type'          : 'Default',
            'Score'         : score,
            'Precision'     : precision,
            'Recall'        : recall,
            'F1'            : f1,
            'Classification': classification,
            'Params'        : None,
            'Target_cols'   : self._input.X_train.columns,
            'Pred_y'        : pred_y
        })

        self._output.results = results

        return True
