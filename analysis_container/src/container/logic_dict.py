#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../')
from logic.abstract_logic import AbstractLogic
from logic.abstract_interface import AbstractInterface
from extention.logger import Logger

# 追加したロジッククラス、ロジックインプットクラス、ロジックアウトクラスをここに追記する
# TODO: import importlib を使用して自動インポートしたい

from logic.analyze.sklearn.sklearn_logic import SklearnLogic
from logic.analyze.sklearn.interface.i_sklearn_input import ISklearnInput
from logic.analyze.sklearn.interface.i_sklearn_output import ISklearnOutput

from logic.analyze.light_gbm.light_gbm_logic import LightGbmLogic
from logic.analyze.light_gbm.interface.i_light_gbm_input import ILightGbmInput
from logic.analyze.light_gbm.interface.i_light_gbm_output import ILightGbmOutput

from logic.data_wrangling.pre_processing.pre_processing_logic import PreProcessingLogic
from logic.data_wrangling.pre_processing.interface.i_pre_processing_input import IPreProcessingInput 
from logic.data_wrangling.pre_processing.interface.i_pre_processing_output import IPreProcessingOutput

class LogicDict:
    """ロジック実行用のキーバリュー格納用クラス

       格納以外に検証とクラス変換機能もあり
    """

    # ロジック実行用のキーバリューのキー
    LOGIC_EXEC_KEY = 'logic'                        # 実行対象logicクラス用のキー
    LOGIC_EXEC_INPUT_KEY = 'input'                  # 実行対象logicのインプットインターフェイス用のキー
    LOGIC_EXEC_INPUT_INSTANCE = 'input_instance'    # 実行対象logicのインプットクラスインスタンス用のキー
    LOGIC_EXEC_OUTPUT_KEY = 'output'                # 実行対象logicのアウトプットインターフェイス用のキー
    LOGIC_EXEC_KEY_MAX_LENGTH = 4                   # ロジック実行用のキーバリューのキーの最大数
    LOGIC_EXEC_KEYS: tuple = (LOGIC_EXEC_KEY, LOGIC_EXEC_INPUT_KEY, LOGIC_EXEC_INPUT_INSTANCE, LOGIC_EXEC_OUTPUT_KEY)
    LOGIC_EXEC_REQUIRED_KEYS: tuple = (LOGIC_EXEC_KEY,)

    __logger = None
    __logic_exec_class_dict: dict

    def __init__(self, logic_exec_dict: dict):
        """コンストラクタ

        Args:
            logic_exec_dict (dict): ロジック実行用のキーバリュー
            logic_exec_dictは下記の形式で格納されている
            [{ 'logic': 'LogicClassName', 'input': 'InputClassName', 'output': 'OutputClassName'}]
        """
        self.__logic_exec_dict: dict = logic_exec_dict

    def set_logger(self, logger: Logger):
        """logger set
        
        Args:
            logger (Logger): Logger
        """
        self.__logger = logger

    def get_logic_exec_class_dict(self) -> dict:
        """ロジック実行用のキーバリュー(クラス格納)を取得  
           キーバリューは下記の形式で格納されている  
           [{ 'logic': LogicClass, 'input': InputClass, 'output': OutputClass}]  
        
        Returns:
            dict: ロジック実行用のキーバリュー(クラス格納)
        """
        return self.__logic_exec_class_dict

    def validate_logic_exec_dict(self) -> bool:
        """ロジック実行用のキーバリューを検証する

        Returns:
            bool: true:検証OK false:検証NG
        """

        if not self.__logic_exec_dict:
            self.__logger.error('logic_exec_dict empty')
            return False

        for logic_exec_list in self.__logic_exec_dict:
            logic_exec_keys: list = []
            for logic_exec_key in logic_exec_list:
                logic_exec_keys.append(logic_exec_key)

            if self.__validate_require_and_define_keys(logic_exec_keys) is False:
                self.__logger.error('logic_exec_dict not reuired key or not define key')
                return False

        if self.__validate_required_value_class_name(self.__logic_exec_dict) is False:
            self.__logger.error('logic_exec_dict not only reuired key')
            return False

        self.__logic_exec_class_dict = self.__convert_value_name_to_class(self.__logic_exec_dict)

        if self.__validate_value_instance_type(self.__logic_exec_class_dict) is False:
            self.__logger.error('logic_exec_dict empty value or different abstract type')
            return False

        return True

    def __validate_require_and_define_keys(self, logic_exec_keys: list) -> bool:
        """必須キー、および、定義されたキーが含まれているか検証
        
        Args:
            logic_exec_keys (list): ロジック実行用のキーリスト
        
        Returns:
            bool: true:検証OK false:検証NG
        """        
        if not logic_exec_keys:
            return False
        elif len(logic_exec_keys) > self.LOGIC_EXEC_KEY_MAX_LENGTH:
            # 指定された要素数以内か
            return False
        elif len(logic_exec_keys) != len(set(logic_exec_keys)):
            # 重複要素がないか
            return False

        required_keys: list = list(self.LOGIC_EXEC_REQUIRED_KEYS)
        
        for key in logic_exec_keys:
            if not key in self.LOGIC_EXEC_KEYS:
                return False

            if key in self.LOGIC_EXEC_REQUIRED_KEYS:
                del required_keys[required_keys.index(key)]

        if not required_keys:
            return True        

        return False

    def __validate_required_value_class_name(self, logic_exec_dict: dict) -> bool:
        """ロジック実行用のキーバリューのクラス名(バリュー)必須チェック
        
        Args:
            logic_exec_dict (dict): ロジック実行用のキーバリュ
        
        Returns:
            bool: true:検証OK false:検証NG
        """        
        for class_name_list in logic_exec_dict:
            if not class_name_list[self.LOGIC_EXEC_KEY]:
                return False

        return True

    def __convert_value_name_to_class(self, logic_exec_dict: dict) -> dict:
        """ロジック実行用のキーバリューをクラス名からクラスに変換して返す
        
        Args:
            logic_exec_dict (dict): [description]
        
        Returns:
            dict: 例) [{ 'logic': LogicClass, 'input': InputClass, 'output': OutputClass}]
        """        
        logic_exec_class_dict = logic_exec_dict.copy()
        for index, logic_exec_list in enumerate(logic_exec_dict):
            logic_exec_class_dict[index][self.LOGIC_EXEC_KEY] = globals()[logic_exec_list[self.LOGIC_EXEC_KEY]]

            if logic_exec_list[self.LOGIC_EXEC_INPUT_KEY]:
                logic_exec_class_dict[index][self.LOGIC_EXEC_INPUT_KEY] = globals()[logic_exec_list[self.LOGIC_EXEC_INPUT_KEY]]
            else:
                logic_exec_class_dict[index][self.LOGIC_EXEC_INPUT_KEY] = None

            if logic_exec_list[self.LOGIC_EXEC_OUTPUT_KEY]:
                logic_exec_class_dict[index][self.LOGIC_EXEC_OUTPUT_KEY] = globals()[logic_exec_list[self.LOGIC_EXEC_OUTPUT_KEY]]
            else:
                logic_exec_class_dict[index][self.LOGIC_EXEC_OUTPUT_KEY] = None

        return logic_exec_class_dict

    def __validate_value_instance_type(self, logic_exec_class_dict: dict) -> bool:
        """ロジック実行用のキーに対する特定の型を継承しているか検証
        
        Args:
            logic_exec_class_dict (dict): [description]
        
        Returns:
            bool: true:検証OK false:検証NG
        """

        for logic_exec_class_list in logic_exec_class_dict:
            if issubclass(type(logic_exec_class_list[self.LOGIC_EXEC_KEY]), AbstractLogic):
                return False

            if not logic_exec_class_list[self.LOGIC_EXEC_INPUT_KEY] is None:
                if issubclass(type(logic_exec_class_list[self.LOGIC_EXEC_INPUT_KEY]), AbstractLogic):
                    return False

            if not logic_exec_class_list[self.LOGIC_EXEC_OUTPUT_KEY] is None:
                if issubclass(type(logic_exec_class_list[self.LOGIC_EXEC_OUTPUT_KEY]), AbstractLogic):
                    return False

        return True
