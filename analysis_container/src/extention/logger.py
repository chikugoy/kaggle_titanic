#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import logging.config
from pathlib import Path
from threading import Lock

sys.path.append('./')
from .singleton import Singleton

# TODO: app.logには日時などが出力されないのでなんとかしたい

class Logger(Singleton):
    """Logger
    
    Args:
        Singleton (Singleton): Singleton継承
    """
    LOG_DIR_NAME = 'log'
    LOG_FILE_NAME = 'app.log'
    LOG_INI_DIR_NAME = 'config'
    LOG_INI_FILE_NAME = 'logger.ini'
    LOG_INI_QUAL_NAME = 'outputLogging'

    __logger = None
    __init_setting_log_flg = False

    @classmethod
    def get_instance(cls) -> Singleton:
        """インスタンス（生成）取得とログ初期設定処理
        
        Returns:
            Singleton: 自身を生成して返す
        """        
        instance: Logger = super().get_instance()
        instance.__init_setting_log()
        return instance

    @classmethod
    def debug(cls, message, *args):
        """debug
        
        Args:
            message (any): message
        """        
        if args:
            cls.__logger.debug(message, args)
        else:
            cls.__logger.debug(message)

    @classmethod
    def info(cls, message, *args):
        """info
        
        Args:
            message (any): message
        """        
        if args:
            cls.__logger.info(message, args)
        else:
            cls.__logger.info(message)

    @classmethod
    def warning(cls, message, *args):
        """warning
        
        Args:
            message (any): message
        """        
        if args:
            cls.__logger.warning(message, args)
        else:
            cls.__logger.warning(message)

    @classmethod
    def error(cls, message, *args):
        """error
        
        Args:
            message (any): message
        """        
        if args:
            cls.__logger.error(message, args)
        else:
            cls.__logger.error(message)

    @classmethod
    def critical(cls, message, *args):
        """critical
        
        Args:
            message (any): message
        """        
        if args:
            cls.__logger.critical(message, args)
        else:
            cls.__logger.critical(message)

    @classmethod
    def __init_setting_log(cls):
        """ログ初期設定処理  
           ログレベルの設定はiniで行う
        """        
        if cls.__init_setting_log_flg is True:
            return

        current_path = str(Path.cwd())

        # TODO: src直下でないと動かないので改善したい
        log_ini_file_path = current_path + os.sep + \
            cls.LOG_INI_DIR_NAME + os.sep + cls.LOG_INI_FILE_NAME
        logging.config.fileConfig(
            log_ini_file_path, disable_existing_loggers=False)

        logger = logging.getLogger(cls.LOG_INI_QUAL_NAME)
        # logger.setLevel(logging.DEBUG)        # ログレベルの設定はiniで行う

        # TODO: src直下でないと動かないので改善したい
        log_file_path = current_path + os.sep + \
            cls.LOG_DIR_NAME + os.sep + cls.LOG_FILE_NAME
        get_handler = logging.FileHandler(log_file_path)
        logger.addHandler(get_handler)

        cls.__logger = logger

        cls.__init_setting_log_flg = True

        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)        

