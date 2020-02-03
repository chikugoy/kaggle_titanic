#!/usr/bin/env python
# -*- coding: utf-8 -*-

import configparser
import os
import errno
import sys
from threading import Lock

sys.path.append('./')
from .singleton import Singleton


class ConfigIni(Singleton):

    INI_FILE_NAME = 'config.ini'        # config.pyと同階層に配置
    INI_ENCODING = 'utf-8'

    INI_SECTION_DEFAULT = 'DEFAULT'
    INI_SECTION_LOG = 'LOG'
    INI_REQUIRED_SECTIONS: tuple = (INI_SECTION_DEFAULT, INI_SECTION_LOG)

    INI_SECTION_LOG_KEY_DEBUG = 'Debug'

    INI_ON_FLG = 'on'
    INI_OFF_FLG = 'off'

    section_items = {}

    @classmethod
    def get_instance(cls):
        instance: ConfigIni = super().get_instance()
        instance.__init_ini()
        return instance

    @classmethod
    def get_item(cls, section, key):
        if not cls.section_items:
            with Lock():
                cls.__init_ini()

        return cls.section_items[section][key]

    @classmethod
    def __init_ini(cls):
        config_ini_parser = cls.__get_config_parser()

        sections = config_ini_parser.sections()

        if cls.__validate_ini_required_sections(sections) is False:
            raise Exception('iniファイルのセクションの指定が不正です')

        cls.section_items = cls.__get_config_ini_section_items(
            config_ini_parser, sections)

        if not cls.section_items:
            raise Exception('iniファイルの設定が不正')

    @classmethod
    def __get_config_parser(cls):
        config_ini_parser = configparser.ConfigParser()
        config_ini_parser_path = cls.INI_FILE_NAME

        if not os.path.exists(config_ini_parser_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(
                errno.ENOENT), config_ini_parser_path)

        config_ini_parser.read(config_ini_parser_path,
                               encoding=cls.INI_ENCODING)
        return config_ini_parser

    @classmethod
    def __validate_ini_required_sections(cls, sections: list):
        if not sections:
            return False

        required_sections: list = list(cls.INI_REQUIRED_SECTIONS)

        for section in sections:
            if not section in required_sections:
                return False

            del required_sections[section]

        if not required_sections:
            return True

        return False

    @classmethod
    def __get_config_ini_section_items(cls, config_ini_parser, sections: list):
        section_items = {}
        for section in sections:
            items = config_ini_parser.items(sections)
            section_items[section] = items

        return section_items
