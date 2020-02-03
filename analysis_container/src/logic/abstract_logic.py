#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from abc import ABCMeta, abstractmethod

sys.path.append('./')
from .abstract_interface import AbstractInterface

sys.path.append('../')
from extention.logger import Logger


class AbstractLogic(metaclass=ABCMeta):
    _logger: Logger = Logger.get_instance()

    def __init__(self, inputValue: AbstractInterface, output: AbstractInterface):
        self._input = inputValue
        self._output = output

    @abstractmethod
    def execute(self) -> bool:
        return True

    def get_output(self) -> AbstractInterface:
        return self._output
