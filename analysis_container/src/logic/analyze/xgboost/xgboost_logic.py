#!/usr/bin/env python
# -*- coding: utf-8 -*-

from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic
from logic.analyze.xgboost.enum.e_xgboost_type import EXGBoostType
from .xgboost_logic_classification import XGBoostLogicClassification
from .xgboost_logic_regression import XGBoostLogicRegression

import sys

sys.path.append('/../../')


class XGBoostLogic(AbstractLogic):

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        if self._input.X_train is None or self._input.Y_train is None:
            raise Exception('X_train or Y_train Nothing')

        ret = False
        if self._input.xgboost_type == EXGBoostType.CLASSIFICATION:
            classification = XGBoostLogicClassification(self._input, self._output)
            ret = classification.execute()
            self._output = classification._output
        else:
            regression = XGBoostLogicRegression(self._input, self._output)
            ret = regression.execute()
            self._output = regression._output

        return ret
