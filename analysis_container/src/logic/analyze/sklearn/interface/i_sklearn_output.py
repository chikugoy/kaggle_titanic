#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('./../../../')
from logic.abstract_interface import AbstractInterface


class ISklearnOutput(AbstractInterface):

    model_name: str
    results: list

    def __init__(self):
        super().__init__()
        pass

