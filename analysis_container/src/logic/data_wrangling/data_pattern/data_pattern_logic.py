#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
import random as rnd
from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic
import sys

sys.path.append('/../../')


class DataPatternLogic(AbstractLogic):

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        super().__init__(inputValue, output)

    def execute(self) -> bool:
        if self._input.X_train is None or self._input.Y_train is None or self._input.target_cols is None:
            raise Exception('X_train or Y_train Nothing')

        self._output = self._input
        drop_cols = []
        for col in self._output.X_train.columns:
            if not col in self._output.target_cols:
                drop_cols.append(col)

        if len(drop_cols) > 0:
            self._output.X_train.drop(drop_cols, axis=1)

        return True
