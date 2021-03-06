#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb

from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic
from logic.analyze.light_gbm.enum.e_light_gbm_type import ELightGbmType
from .light_gbm_logic_classification import LightGbmLogicClassification
from .light_gbm_logic_regression import LightGbmLogicRegression

sys.path.append('/../../')


class LightGbmLogic(AbstractLogic):

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        if self._input.X_train is None or self._input.Y_train is None:
            raise Exception('X_train or Y_train Nothing')

        ret = False
        if self._input.light_gbm_type == ELightGbmType.CLASSIFICATION:
            classification = LightGbmLogicClassification(self._input, self._output)
            ret = classification.execute()
            self._output = classification._output
        else:
            regression = LightGbmLogicRegression(self._input, self._output)
            ret = regression.execute()
            self._output = regression._output

        return ret
