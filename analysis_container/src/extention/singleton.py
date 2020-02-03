#!/usr/bin/env python
# -*- coding: utf-8 -*-

from threading import Lock


class Singleton:
    """Singleton
    
    Raises:
        NotImplementedError: newした場合の例外
    """
    __unique_instance = None
    __lock = Lock()  # クラスロック

    def __new__(cls):
        """new禁止用処理
        
        Raises:
            NotImplementedError: newした場合の例外
        """        
        raise NotImplementedError('Cannot initialize via Constructor')

    @classmethod
    def get_instance(cls) -> object:
        """インスタンス(生成)取得
        
        Returns:
            Singleton: 自身を生成して返す
        """        
        if not cls.__unique_instance:
            with cls.__lock:
                if not cls.__unique_instance:
                    cls.__unique_instance = super().__new__(cls)
        return cls.__unique_instance
