#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('./../../../')
from logic.abstract_interface import AbstractInterface


class IPreProcessingInput(AbstractInterface):

    input_train_path: str
    input_test_path: str
    output_x_train_path: str
    output_y_train_path: str
    output_x_test_path: str

    model: any
    cv_value: int
    X_train: any
    Y_train: any
    grid_search_params: dict
    random_search_params: dict


    def __init__(self):
        super().__init__()
        pass
