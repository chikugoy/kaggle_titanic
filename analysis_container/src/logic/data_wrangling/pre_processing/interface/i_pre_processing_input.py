#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('./../../../')
from logic.abstract_interface import AbstractInterface


class IPreProcessingInput(AbstractInterface):

    input_train_path: str = None
    input_test_path: str = None
    output_x_train_path: str = None
    output_y_train_path: str = None
    output_x_test_path: str = None

    def __init__(self):
        super().__init__()
        pass
