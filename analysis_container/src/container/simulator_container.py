#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import concurrent.futures

sys.path.append('./')
from .logic_dict import LogicDict
from .abstract_container import AbstractContainer

sys.path.append('../')
from logic.abstract_interface import AbstractInterface
from logic.abstract_logic import AbstractLogic


class SimulatorContainer(AbstractContainer):
    """Simulator Container

    Args:
        AbstractContainer (AbstractContainer): AbstractContainer Inheritance
    """
    current_logic_output: AbstractInterface = None

    def __init__(self, logic_dict: LogicDict, options={}):
        # TODO: 並列実行option
        """constructor

        Args:
            logic_dict (LogicDict): [description]
            options ([type]): [description]
        """
        self.__logic_dict: LogicDict = logic_dict

    def execute(self) -> bool:
        """container execute

        Returns:
            bool: True: execute success False: execute fail
        """
        self._logger.debug('container execute start >>>>>>>>>>>>>>>>>>>')
        try:
            self.__logic_dict.set_logger(self._logger)
            if not self.__logic_dict.validate_logic_exec_dict():
                self._logger.error('logic validation return False')
                return False


            if not self.__execute_logic_by_dependency_injection(self.__logic_dict.get_logic_exec_class_dict()):
                self._logger.error('logic execute return False')
                return False

        except Exception as e:
            self._logger.error('container execute catch Exception:', e)
            return False
        else:
            self._logger.debug('container execute success')
            return True
        finally:
            self._logger.debug('container execute end  <<<<<<<<<<<<<<<<<<<<')

    def __execute_logic_by_dependency_injection(self, logic_exec_class_dict: dict) -> bool:
        """logic classをDIして実行

        Args:
            logic_exec_class_dict (dict): 実行対象ロジックdict

        Returns:
            bool: True: logic execute success False: logic execute fail
        """

        for logic_exec_class_list in logic_exec_class_dict:
            # logic class Dependency Injection
            logic_class: AbstractLogic = logic_exec_class_list[LogicDict.LOGIC_EXEC_KEY]

            logic_input_instance: AbstractInterface = None
            logic_input_class_name = 'nothing'
            if logic_exec_class_list[LogicDict.LOGIC_EXEC_INPUT_KEY]:

                if (LogicDict.LOGIC_EXEC_INPUT_INSTANCE in logic_exec_class_list and
                    logic_exec_class_list[LogicDict.LOGIC_EXEC_INPUT_INSTANCE]):
                    logic_input_instance = logic_exec_class_list[LogicDict.LOGIC_EXEC_INPUT_INSTANCE]
                else:
                    logic_input_instance = logic_exec_class_list[LogicDict.LOGIC_EXEC_INPUT_KEY]()

                logic_input_class_name = logic_input_instance.__class__.__name__

                for logic_output in self._logic_outputs:
                    if logic_output.__class__.__name__ == logic_input_class_name:
                        logic_input_instance = logic_output

            logic_output_instance: AbstractInterface = None
            logic_output_class_name = 'nothing'
            if logic_exec_class_list[LogicDict.LOGIC_EXEC_OUTPUT_KEY]:
                logic_output_instance = logic_exec_class_list[LogicDict.LOGIC_EXEC_OUTPUT_KEY]()
                logic_output_class_name = logic_output_instance.__class__.__name__

            logic_instance: AbstractLogic = logic_class(
                logic_input_instance, logic_output_instance)

            self._logger.debug('logic start >>>>>>>>>>>')
            self._logger.debug('logic:' + logic_instance.__class__.__name__ +
                               ' input:' + logic_input_class_name +
                               ' output:' + logic_output_class_name)

            if logic_instance.execute() is False:
                self._logger.error('logic execute fail')
                return False
            else:
                self._logger.debug('logic execute success')

            self._current_logic_output = logic_instance.get_output()

            is_set_logic_output:bool = False
            for i, logic_output in enumerate(self._logic_outputs):
                if self._current_logic_output.__class__.__name__ == logic_output.__class__.__name__:
                    self._logic_outputs[i] = self._current_logic_output
                    is_set_logic_output = True

            if is_set_logic_output == False:
                self._logic_outputs.append(self._current_logic_output)

            self._logger.debug('logic end  <<<<<<<<<<<')

        return True
