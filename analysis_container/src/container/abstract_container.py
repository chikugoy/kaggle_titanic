#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from abc import ABCMeta, abstractmethod

sys.path.append('../')
from extention.logger import Logger
from logic.abstract_interface import AbstractInterface


class AbstractContainer(metaclass=ABCMeta):
    """Container Abstract Class
       ここでのコンテナはpython的なlistやdictではなく
       ロジック実行用の箱の事

    Args:
        metaclass (ABCMeta, optional): Defaults to ABCMeta.
    """

    _logger: Logger = Logger.get_instance()
    _current_logic_output: AbstractInterface

    def __init__(self):
        """constructor
        """
        pass

    @abstractmethod
    def execute(self) -> bool:
        return True

    def get_logic_output(self) -> AbstractInterface:
        return self._current_logic_output
