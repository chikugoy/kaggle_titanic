#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('./../../../')
from logic.abstract_interface import AbstractInterface


class IPreProcessingOutput(AbstractInterface):

    x_train_path: str = None
    y_train_path: str = None
    x_test_path: str = None
    X_train: any = None
    Y_train: any = None
    X_test: any = None

    def __init__(self):
        super().__init__()
        pass

