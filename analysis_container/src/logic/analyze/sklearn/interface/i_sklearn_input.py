#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('./../../../')
from logic.abstract_interface import AbstractInterface


class ISklearnInput(AbstractInterface):

    cv_value: int
    X_train: any
    Y_train: any
    grid_search_params: dict
    random_search_params: dict


    def __init__(self):
        super().__init__()
        pass
